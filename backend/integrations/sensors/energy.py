# app/integrations/sensors/energy.py

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Protocol

logger = logging.getLogger(__name__)


class EnergyStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HIGH_CONSUMPTION = "high_consumption"
    LOW_POWER_FACTOR = "low_power_factor"
    OVERLOAD = "overload"
    SENSOR_FAULT = "sensor_fault"


class EnergyQuality(str, Enum):
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"


@dataclass(slots=True)
class EnergySensorConfig:
    sensor_id: str
    asset_id: str
    site_id: str

    sample_interval: float = 1.0

    nominal_voltage_v: float = 415.0
    rated_power_kw: float = 75.0

    nominal_power_factor: float = 0.88

    warning_power_kw: float = 65.0
    critical_power_kw: float = 80.0

    warning_power_factor: float = 0.85
    critical_power_factor: float = 0.70

    energy_warning_kwh: float | None = None

    electricity_cost_per_kwh: float = 8.0

    calibration_scale: float = 1.0
    calibration_offset: float = 0.0


@dataclass(slots=True)
class EnergyReading:
    sensor_id: str
    asset_id: str
    site_id: str

    voltage_v: float
    current_a: float

    active_power_kw: float
    apparent_power_kva: float
    reactive_power_kvar: float

    power_factor: float

    energy_kwh: float
    estimated_cost: float

    load_percent: float

    status: EnergyStatus
    quality: EnergyQuality

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "asset_id": self.asset_id,
            "site_id": self.site_id,
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
            "active_power_kw": self.active_power_kw,
            "apparent_power_kva": self.apparent_power_kva,
            "reactive_power_kvar": self.reactive_power_kvar,
            "power_factor": self.power_factor,
            "energy_kwh": self.energy_kwh,
            "estimated_cost": self.estimated_cost,
            "load_percent": self.load_percent,
            "status": self.status.value,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EnergySample:
    voltage_v: float
    current_a: float
    power_factor: float


class EnergySource(Protocol):
    async def read_energy_sample(
        self,
    ) -> EnergySample:
        ...


class EnergySensor:
    """
    Industrial energy-monitoring sensor.

    Provides:
    - voltage/current monitoring
    - active power
    - apparent power
    - reactive power
    - power factor
    - accumulated kWh
    - load percentage
    - energy-cost estimation
    - overload detection
    - low-power-factor detection
    - async telemetry streaming
    """

    def __init__(
        self,
        config: EnergySensorConfig,
        source: EnergySource,
    ) -> None:
        self.config = config
        self.source = source

        self._energy_kwh = 0.0

        self._last_timestamp: datetime | None = None
        self._last_reading: EnergyReading | None = None

    async def read(self) -> EnergyReading:
        now = datetime.now(timezone.utc)

        try:
            sample = (
                await self.source.read_energy_sample()
            )

            voltage = self._calibrate_voltage(
                sample.voltage_v
            )

            current = max(
                0.0,
                float(sample.current_a),
            )

            power_factor = self._clamp(
                float(sample.power_factor),
                0.0,
                1.0,
            )

            apparent_power = (
                calculate_three_phase_apparent_power_kva(
                    voltage_v=voltage,
                    current_a=current,
                )
            )

            active_power = (
                apparent_power * power_factor
            )

            reactive_power = (
                calculate_reactive_power_kvar(
                    apparent_power_kva=apparent_power,
                    active_power_kw=active_power,
                )
            )

            self._integrate_energy(
                active_power,
                now,
            )

            load_percent = (
                calculate_load_percent(
                    active_power_kw=active_power,
                    rated_power_kw=(
                        self.config.rated_power_kw
                    ),
                )
            )

            quality = self._evaluate_quality(
                voltage=voltage,
                current=current,
                power_factor=power_factor,
            )

            status = self._evaluate_status(
                active_power_kw=active_power,
                power_factor=power_factor,
                load_percent=load_percent,
                quality=quality,
            )

            estimated_cost = (
                self._energy_kwh
                * self.config.electricity_cost_per_kwh
            )

            reading = EnergyReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                voltage_v=round(voltage, 3),
                current_a=round(current, 3),
                active_power_kw=round(
                    active_power,
                    4,
                ),
                apparent_power_kva=round(
                    apparent_power,
                    4,
                ),
                reactive_power_kvar=round(
                    reactive_power,
                    4,
                ),
                power_factor=round(
                    power_factor,
                    4,
                ),
                energy_kwh=round(
                    self._energy_kwh,
                    6,
                ),
                estimated_cost=round(
                    estimated_cost,
                    4,
                ),
                load_percent=round(
                    load_percent,
                    3,
                ),
                status=status,
                quality=quality,
                timestamp=now,
                metadata={
                    "rated_power_kw": (
                        self.config.rated_power_kw
                    ),
                    "cost_per_kwh": (
                        self.config.electricity_cost_per_kwh
                    ),
                },
            )

            self._last_reading = reading

            return reading

        except Exception as exc:
            logger.exception(
                "Energy sensor read failed sensor=%s",
                self.config.sensor_id,
            )

            reading = EnergyReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                voltage_v=0.0,
                current_a=0.0,
                active_power_kw=0.0,
                apparent_power_kva=0.0,
                reactive_power_kvar=0.0,
                power_factor=0.0,
                energy_kwh=round(
                    self._energy_kwh,
                    6,
                ),
                estimated_cost=round(
                    self._energy_kwh
                    * self.config.electricity_cost_per_kwh,
                    4,
                ),
                load_percent=0.0,
                status=EnergyStatus.SENSOR_FAULT,
                quality=EnergyQuality.BAD,
                timestamp=now,
                metadata={
                    "error": str(exc),
                },
            )

            self._last_reading = reading

            return reading

    # --------------------------------------------------
    # Energy integration
    # --------------------------------------------------

    def _integrate_energy(
        self,
        active_power_kw: float,
        timestamp: datetime,
    ) -> None:
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return

        elapsed_seconds = (
            timestamp - self._last_timestamp
        ).total_seconds()

        self._last_timestamp = timestamp

        if elapsed_seconds <= 0:
            return

        # Protect against large integration jumps
        # after a long sensor/network outage.
        max_interval = max(
            60.0,
            self.config.sample_interval * 10.0,
        )

        elapsed_seconds = min(
            elapsed_seconds,
            max_interval,
        )

        hours = elapsed_seconds / 3600.0

        self._energy_kwh += (
            max(0.0, active_power_kw)
            * hours
        )

    # --------------------------------------------------
    # Calibration / quality
    # --------------------------------------------------

    def _calibrate_voltage(
        self,
        voltage: float,
    ) -> float:
        value = float(voltage)

        return (
            value
            * self.config.calibration_scale
            + self.config.calibration_offset
        )

    def _evaluate_quality(
        self,
        *,
        voltage: float,
        current: float,
        power_factor: float,
    ) -> EnergyQuality:
        values = (
            voltage,
            current,
            power_factor,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            return EnergyQuality.BAD

        if voltage < 0 or current < 0:
            return EnergyQuality.BAD

        nominal = (
            self.config.nominal_voltage_v
        )

        if nominal <= 0:
            return EnergyQuality.BAD

        deviation = abs(
            voltage - nominal
        ) / nominal

        if deviation > 0.20:
            return EnergyQuality.UNCERTAIN

        return EnergyQuality.GOOD

    # --------------------------------------------------
    # Status
    # --------------------------------------------------

    def _evaluate_status(
        self,
        *,
        active_power_kw: float,
        power_factor: float,
        load_percent: float,
        quality: EnergyQuality,
    ) -> EnergyStatus:
        if quality == EnergyQuality.BAD:
            return EnergyStatus.SENSOR_FAULT

        if (
            active_power_kw
            >= self.config.critical_power_kw
            or load_percent >= 110.0
        ):
            return EnergyStatus.OVERLOAD

        if (
            power_factor
            <= self.config.critical_power_factor
        ):
            return EnergyStatus.CRITICAL

        if (
            active_power_kw
            >= self.config.warning_power_kw
        ):
            return EnergyStatus.HIGH_CONSUMPTION

        if (
            power_factor
            <= self.config.warning_power_factor
        ):
            return EnergyStatus.LOW_POWER_FACTOR

        if (
            self.config.energy_warning_kwh
            is not None
            and self._energy_kwh
            >= self.config.energy_warning_kwh
        ):
            return EnergyStatus.WARNING

        return EnergyStatus.NORMAL

    # --------------------------------------------------
    # Streaming
    # --------------------------------------------------

    async def stream(
        self,
        *,
        interval: float | None = None,
    ) -> AsyncIterator[EnergyReading]:
        delay = (
            interval
            if interval is not None
            else self.config.sample_interval
        )

        if delay <= 0:
            raise ValueError(
                "Energy sensor interval must be positive"
            )

        while True:
            yield await self.read()

            await asyncio.sleep(delay)

    # --------------------------------------------------
    # State
    # --------------------------------------------------

    @property
    def energy_kwh(self) -> float:
        return self._energy_kwh

    @property
    def last_reading(
        self,
    ) -> EnergyReading | None:
        return self._last_reading

    def reset_energy(
        self,
        value_kwh: float = 0.0,
    ) -> None:
        if value_kwh < 0:
            raise ValueError(
                "Energy value cannot be negative"
            )

        self._energy_kwh = float(
            value_kwh
        )

        self._last_timestamp = None

    def set_energy_kwh(
        self,
        value: float,
    ) -> None:
        self.reset_energy(value)

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )


# ---------------------------------------------------------
# Simulated source
# ---------------------------------------------------------

class SimulatedEnergySource:
    """
    Simulated industrial three-phase energy source.

    Useful for:
    - development
    - demo dashboards
    - predictive-maintenance tests
    - energy anomaly testing
    """

    def __init__(
        self,
        *,
        nominal_voltage_v: float = 415.0,
        rated_current_a: float = 100.0,
        nominal_power_factor: float = 0.88,
        seed: int | None = None,
    ) -> None:
        self.nominal_voltage_v = (
            nominal_voltage_v
        )

        self.rated_current_a = (
            rated_current_a
        )

        self.nominal_power_factor = (
            nominal_power_factor
        )

        self.load_percent = 70.0
        self.degradation = 0.0

        self._overload = False
        self._low_power_factor = False
        self._voltage_sag = False
        self._voltage_swell = False
        self._disconnected = False

        self._rng = random.Random(seed)

    async def read_energy_sample(
        self,
    ) -> EnergySample:
        await asyncio.sleep(0)

        if self._disconnected:
            raise ConnectionError(
                "Energy sensor disconnected"
            )

        load = max(
            0.0,
            self.load_percent / 100.0,
        )

        voltage = (
            self.nominal_voltage_v
            + self._rng.uniform(
                -3.0,
                3.0,
            )
        )

        if self._voltage_sag:
            voltage *= 0.72

        if self._voltage_swell:
            voltage *= 1.20

        current = (
            self.rated_current_a
            * load
            + self._rng.uniform(
                -1.5,
                1.5,
            )
        )

        current += (
            self.degradation
            * self.rated_current_a
            * 0.12
        )

        if self._overload:
            current += (
                self.rated_current_a
                * 0.60
            )

        power_factor = (
            self.nominal_power_factor
            - self.degradation * 0.08
            + self._rng.uniform(
                -0.01,
                0.01,
            )
        )

        if self._low_power_factor:
            power_factor = self._rng.uniform(
                0.55,
                0.68,
            )

        power_factor = max(
            0.0,
            min(
                1.0,
                power_factor,
            ),
        )

        return EnergySample(
            voltage_v=max(
                0.0,
                voltage,
            ),
            current_a=max(
                0.0,
                current,
            ),
            power_factor=power_factor,
        )

    def set_load(
        self,
        percent: float,
    ) -> None:
        self.load_percent = max(
            0.0,
            min(
                160.0,
                float(percent),
            ),
        )

    def set_degradation(
        self,
        value: float,
    ) -> None:
        self.degradation = max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    def inject_overload(self) -> None:
        self._overload = True

    def inject_low_power_factor(self) -> None:
        self._low_power_factor = True

    def inject_voltage_sag(self) -> None:
        self._voltage_sag = True
        self._voltage_swell = False

    def inject_voltage_swell(self) -> None:
        self._voltage_swell = True
        self._voltage_sag = False

    def disconnect(self) -> None:
        self._disconnected = True

    def clear_faults(self) -> None:
        self._overload = False
        self._low_power_factor = False
        self._voltage_sag = False
        self._voltage_swell = False
        self._disconnected = False


# ---------------------------------------------------------
# Electrical calculations
# ---------------------------------------------------------

def calculate_three_phase_active_power_kw(
    *,
    voltage_v: float,
    current_a: float,
    power_factor: float,
) -> float:
    if voltage_v < 0:
        raise ValueError(
            "voltage_v cannot be negative"
        )

    if current_a < 0:
        raise ValueError(
            "current_a cannot be negative"
        )

    pf = max(
        0.0,
        min(
            1.0,
            float(power_factor),
        ),
    )

    watts = (
        math.sqrt(3.0)
        * voltage_v
        * current_a
        * pf
    )

    return watts / 1000.0


def calculate_three_phase_apparent_power_kva(
    *,
    voltage_v: float,
    current_a: float,
) -> float:
    if voltage_v < 0 or current_a < 0:
        raise ValueError(
            "Voltage/current cannot be negative"
        )

    return (
        math.sqrt(3.0)
        * voltage_v
        * current_a
        / 1000.0
    )


def calculate_reactive_power_kvar(
    *,
    apparent_power_kva: float,
    active_power_kw: float,
) -> float:
    if apparent_power_kva < 0:
        raise ValueError(
            "Apparent power cannot be negative"
        )

    if active_power_kw < 0:
        raise ValueError(
            "Active power cannot be negative"
        )

    difference = max(
        0.0,
        apparent_power_kva ** 2
        - active_power_kw ** 2,
    )

    return math.sqrt(difference)


def calculate_load_percent(
    *,
    active_power_kw: float,
    rated_power_kw: float,
) -> float:
    if rated_power_kw <= 0:
        raise ValueError(
            "rated_power_kw must be greater than zero"
        )

    return max(
        0.0,
        active_power_kw
        / rated_power_kw
        * 100.0,
    )


def calculate_energy_cost(
    *,
    energy_kwh: float,
    cost_per_kwh: float,
) -> float:
    if energy_kwh < 0:
        raise ValueError(
            "energy_kwh cannot be negative"
        )

    if cost_per_kwh < 0:
        raise ValueError(
            "cost_per_kwh cannot be negative"
        )

    return energy_kwh * cost_per_kwh


def calculate_specific_energy_consumption(
    *,
    energy_kwh: float,
    production_units: int,
) -> float | None:
    """
    Energy efficiency KPI:

        kWh / produced unit
    """

    if production_units <= 0:
        return None

    return (
        energy_kwh
        / production_units
    )


def calculate_energy_efficiency(
    *,
    useful_output_kw: float,
    input_power_kw: float,
) -> float:
    if input_power_kw <= 0:
        return 0.0

    efficiency = (
        useful_output_kw
        / input_power_kw
        * 100.0
    )

    return max(
        0.0,
        min(
            100.0,
            efficiency,
        ),
    )


def create_energy_sensor(
    *,
    sensor_id: str,
    asset_id: str,
    site_id: str,
    rated_power_kw: float = 75.0,
    nominal_voltage_v: float = 415.0,
    rated_current_a: float = 100.0,
    electricity_cost_per_kwh: float = 8.0,
    seed: int | None = None,
) -> EnergySensor:
    source = SimulatedEnergySource(
        nominal_voltage_v=nominal_voltage_v,
        rated_current_a=rated_current_a,
        seed=seed,
    )

    config = EnergySensorConfig(
        sensor_id=sensor_id,
        asset_id=asset_id,
        site_id=site_id,
        nominal_voltage_v=nominal_voltage_v,
        rated_power_kw=rated_power_kw,
        electricity_cost_per_kwh=(
            electricity_cost_per_kwh
        ),
    )

    return EnergySensor(
        config=config,
        source=source,
    )