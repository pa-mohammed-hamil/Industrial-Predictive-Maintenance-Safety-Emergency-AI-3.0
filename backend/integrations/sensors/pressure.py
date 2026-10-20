# app/integrations/sensors/pressure.py

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


class PressureStatus(str, Enum):
    NORMAL = "normal"
    WARNING_LOW = "warning_low"
    WARNING_HIGH = "warning_high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"
    PRESSURE_DROP = "pressure_drop"
    PRESSURE_SPIKE = "pressure_spike"
    SENSOR_FAULT = "sensor_fault"


class PressureQuality(str, Enum):
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"


@dataclass(slots=True)
class PressureSensorConfig:
    sensor_id: str
    asset_id: str
    site_id: str

    location: str = "general"
    unit: str = "bar"

    sample_interval: float = 1.0

    nominal_pressure: float = 5.0

    minimum_pressure: float = 0.0
    maximum_pressure: float = 20.0

    warning_low: float = 2.0
    warning_high: float = 8.0

    critical_low: float = 1.0
    critical_high: float = 10.0

    pressure_drop_threshold: float = 1.5
    pressure_spike_threshold: float = 2.0

    calibration_scale: float = 1.0
    calibration_offset: float = 0.0


@dataclass(slots=True)
class PressureReading:
    sensor_id: str
    asset_id: str
    site_id: str

    pressure: float
    unit: str
    location: str

    status: PressureStatus
    quality: PressureQuality

    pressure_change: float = 0.0
    change_rate: float = 0.0

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
            "pressure": self.pressure,
            "unit": self.unit,
            "location": self.location,
            "status": self.status.value,
            "quality": self.quality.value,
            "pressure_change": self.pressure_change,
            "change_rate": self.change_rate,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class PressureSource(Protocol):
    async def read_pressure(self) -> float:
        ...


class PressureSensor:
    """
    Industrial pressure sensor abstraction.

    Supports:
    - calibrated pressure acquisition
    - low/high pressure detection
    - critical pressure detection
    - sudden pressure-drop detection
    - pressure-spike detection
    - rate-of-change monitoring
    - sensor quality validation
    - async streaming
    """

    def __init__(
        self,
        config: PressureSensorConfig,
        source: PressureSource,
    ) -> None:
        self.config = config
        self.source = source

        self._last_value: float | None = None
        self._last_timestamp: datetime | None = None
        self._last_reading: PressureReading | None = None

    async def read(self) -> PressureReading:
        now = datetime.now(timezone.utc)

        try:
            raw_value = await self.source.read_pressure()

            pressure = self._calibrate(raw_value)

            quality = self._evaluate_quality(
                pressure
            )

            pressure_change = 0.0
            change_rate = 0.0

            if self._last_value is not None:
                pressure_change = (
                    pressure - self._last_value
                )

            if self._last_timestamp is not None:
                elapsed = (
                    now - self._last_timestamp
                ).total_seconds()

                if elapsed > 0:
                    change_rate = (
                        pressure_change / elapsed
                    )

            status = self._evaluate_status(
                pressure=pressure,
                pressure_change=pressure_change,
                quality=quality,
            )

            reading = PressureReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                pressure=round(pressure, 4),
                unit=self.config.unit,
                location=self.config.location,
                status=status,
                quality=quality,
                pressure_change=round(
                    pressure_change,
                    4,
                ),
                change_rate=round(
                    change_rate,
                    4,
                ),
                timestamp=now,
                metadata={
                    "raw_pressure": raw_value,
                    "nominal_pressure": (
                        self.config.nominal_pressure
                    ),
                },
            )

            self._last_value = pressure
            self._last_timestamp = now
            self._last_reading = reading

            return reading

        except Exception as exc:
            logger.exception(
                "Pressure sensor read failed "
                "sensor=%s",
                self.config.sensor_id,
            )

            reading = PressureReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                pressure=0.0,
                unit=self.config.unit,
                location=self.config.location,
                status=(
                    PressureStatus.SENSOR_FAULT
                ),
                quality=PressureQuality.BAD,
                timestamp=now,
                metadata={
                    "error": str(exc),
                },
            )

            self._last_reading = reading

            return reading

    # --------------------------------------------------
    # Calibration
    # --------------------------------------------------

    def _calibrate(
        self,
        value: float,
    ) -> float:
        numeric = float(value)

        return (
            numeric
            * self.config.calibration_scale
            + self.config.calibration_offset
        )

    # --------------------------------------------------
    # Quality
    # --------------------------------------------------

    def _evaluate_quality(
        self,
        pressure: float,
    ) -> PressureQuality:
        if not math.isfinite(pressure):
            return PressureQuality.BAD

        if pressure < 0:
            return PressureQuality.BAD

        if (
            pressure < self.config.minimum_pressure
            or pressure > self.config.maximum_pressure
        ):
            return PressureQuality.UNCERTAIN

        return PressureQuality.GOOD

    # --------------------------------------------------
    # Threshold evaluation
    # --------------------------------------------------

    def _evaluate_status(
        self,
        *,
        pressure: float,
        pressure_change: float,
        quality: PressureQuality,
    ) -> PressureStatus:
        if quality == PressureQuality.BAD:
            return PressureStatus.SENSOR_FAULT

        if (
            self._last_value is not None
            and pressure_change
            <= -abs(
                self.config.pressure_drop_threshold
            )
        ):
            return PressureStatus.PRESSURE_DROP

        if (
            self._last_value is not None
            and pressure_change
            >= abs(
                self.config.pressure_spike_threshold
            )
        ):
            return PressureStatus.PRESSURE_SPIKE

        if pressure <= self.config.critical_low:
            return PressureStatus.CRITICAL_LOW

        if pressure >= self.config.critical_high:
            return PressureStatus.CRITICAL_HIGH

        if pressure <= self.config.warning_low:
            return PressureStatus.WARNING_LOW

        if pressure >= self.config.warning_high:
            return PressureStatus.WARNING_HIGH

        return PressureStatus.NORMAL

    # --------------------------------------------------
    # Streaming
    # --------------------------------------------------

    async def stream(
        self,
        *,
        interval: float | None = None,
    ) -> AsyncIterator[PressureReading]:
        delay = (
            interval
            if interval is not None
            else self.config.sample_interval
        )

        if delay <= 0:
            raise ValueError(
                "Pressure sensor interval "
                "must be positive"
            )

        while True:
            yield await self.read()

            await asyncio.sleep(delay)

    # --------------------------------------------------
    # State
    # --------------------------------------------------

    @property
    def last_reading(
        self,
    ) -> PressureReading | None:
        return self._last_reading

    @property
    def last_pressure(
        self,
    ) -> float | None:
        return self._last_value

    def reset_history(self) -> None:
        self._last_value = None
        self._last_timestamp = None
        self._last_reading = None


# ---------------------------------------------------------
# Simulated source
# ---------------------------------------------------------

class SimulatedPressureSource:
    """
    Simulated industrial pressure sensor.

    Supports:
    - load-dependent pressure
    - gradual degradation
    - pressure leaks
    - pressure drops
    - pressure spikes
    - blocked-line simulation
    - sensor disconnection
    """

    def __init__(
        self,
        *,
        nominal_pressure: float = 5.0,
        noise: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.nominal_pressure = (
            nominal_pressure
        )

        self.noise = max(
            0.0,
            noise,
        )

        self.load_percent = 70.0
        self.degradation = 0.0

        self._leak = False
        self._pressure_drop = False
        self._pressure_spike = False
        self._blocked_line = False
        self._disconnected = False

        self._rng = random.Random(seed)

    async def read_pressure(self) -> float:
        await asyncio.sleep(0)

        if self._disconnected:
            raise ConnectionError(
                "Pressure sensor disconnected"
            )

        load_factor = max(
            0.0,
            self.load_percent / 100.0,
        )

        pressure = (
            self.nominal_pressure
            * (
                0.65
                + 0.5 * load_factor
            )
        )

        pressure -= (
            self.degradation
            * self.nominal_pressure
            * 0.15
        )

        pressure += self._rng.uniform(
            -self.noise,
            self.noise,
        )

        if self._leak:
            pressure -= (
                self.nominal_pressure
                * 0.35
            )

        if self._pressure_drop:
            pressure -= (
                self.nominal_pressure
                * 0.65
            )

        if self._pressure_spike:
            pressure += (
                self.nominal_pressure
                * 1.2
            )

        if self._blocked_line:
            pressure += (
                self.nominal_pressure
                * 0.9
            )

        return max(
            0.0,
            pressure,
        )

    def set_load(
        self,
        percent: float,
    ) -> None:
        self.load_percent = max(
            0.0,
            min(
                150.0,
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

    def inject_leak(self) -> None:
        self._leak = True

    def inject_pressure_drop(self) -> None:
        self._pressure_drop = True
        self._pressure_spike = False

    def inject_pressure_spike(self) -> None:
        self._pressure_spike = True
        self._pressure_drop = False

    def inject_blocked_line(self) -> None:
        self._blocked_line = True

    def disconnect(self) -> None:
        self._disconnected = True

    def clear_faults(self) -> None:
        self._leak = False
        self._pressure_drop = False
        self._pressure_spike = False
        self._blocked_line = False
        self._disconnected = False


# ---------------------------------------------------------
# Pressure calculations
# ---------------------------------------------------------

def pressure_differential(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
) -> float:
    """Return outlet - inlet pressure."""

    return (
        float(outlet_pressure)
        - float(inlet_pressure)
    )


def pressure_drop_percent(
    *,
    reference_pressure: float,
    measured_pressure: float,
) -> float:
    if reference_pressure <= 0:
        raise ValueError(
            "reference_pressure must be "
            "greater than zero"
        )

    drop = (
        reference_pressure
        - measured_pressure
    )

    return max(
        0.0,
        drop
        / reference_pressure
        * 100.0,
    )


def pressure_deviation_percent(
    *,
    measured_pressure: float,
    nominal_pressure: float,
) -> float:
    if nominal_pressure <= 0:
        raise ValueError(
            "nominal_pressure must be "
            "greater than zero"
        )

    return (
        measured_pressure
        - nominal_pressure
    ) / nominal_pressure * 100.0


def detect_pressure_leak(
    *,
    expected_pressure: float,
    measured_pressure: float,
    tolerance_percent: float = 20.0,
) -> bool:
    if expected_pressure <= 0:
        return False

    drop = pressure_drop_percent(
        reference_pressure=expected_pressure,
        measured_pressure=measured_pressure,
    )

    return drop >= tolerance_percent


def calculate_pressure_health_score(
    *,
    measured_pressure: float,
    nominal_pressure: float,
    critical_deviation_percent: float = 50.0,
) -> float:
    """
    Simple 0-100 pressure health score.

    100 = near nominal pressure.
    0   = critical deviation.
    """

    if nominal_pressure <= 0:
        return 0.0

    deviation = abs(
        pressure_deviation_percent(
            measured_pressure=measured_pressure,
            nominal_pressure=nominal_pressure,
        )
    )

    if critical_deviation_percent <= 0:
        return 0.0

    score = (
        100.0
        * (
            1.0
            - deviation
            / critical_deviation_percent
        )
    )

    return max(
        0.0,
        min(
            100.0,
            score,
        ),
    )


# ---------------------------------------------------------
# Factory
# ---------------------------------------------------------

def create_pressure_sensor(
    *,
    sensor_id: str,
    asset_id: str,
    site_id: str,
    location: str = "oil_system",
    nominal_pressure: float = 5.0,
    seed: int | None = None,
) -> PressureSensor:
    source = SimulatedPressureSource(
        nominal_pressure=nominal_pressure,
        seed=seed,
    )

    config = PressureSensorConfig(
        sensor_id=sensor_id,
        asset_id=asset_id,
        site_id=site_id,
        location=location,
        nominal_pressure=nominal_pressure,
    )

    return PressureSensor(
        config=config,
        source=source,
    )