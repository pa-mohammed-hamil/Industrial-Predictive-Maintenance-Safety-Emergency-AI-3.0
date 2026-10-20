# app/integrations/sensors/current.py

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


class CurrentStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERCURRENT = "overcurrent"
    UNDERCURRENT = "undercurrent"
    SENSOR_FAULT = "sensor_fault"


class CurrentQuality(str, Enum):
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"


@dataclass(slots=True)
class CurrentSensorConfig:
    sensor_id: str
    asset_id: str
    site_id: str

    phase: str = "L1"
    unit: str = "A"

    sample_interval: float = 1.0

    minimum_current: float = 0.0
    maximum_current: float = 150.0

    warning_low: float = 5.0
    warning_high: float = 85.0

    critical_low: float = 1.0
    critical_high: float = 110.0

    nominal_current: float = 55.0

    calibration_offset: float = 0.0
    calibration_scale: float = 1.0


@dataclass(slots=True)
class CurrentReading:
    sensor_id: str
    asset_id: str
    site_id: str

    current_a: float
    phase: str

    status: CurrentStatus
    quality: CurrentQuality

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
            "current_a": self.current_a,
            "phase": self.phase,
            "status": self.status.value,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class CurrentSource(Protocol):
    async def read_current(self) -> float:
        ...


class CurrentSensor:
    """
    Industrial electrical-current sensor integration.

    Supports:
    - async current acquisition
    - calibration
    - threshold classification
    - overcurrent detection
    - undercurrent detection
    - sensor fault detection
    - streaming
    - RMS calculation
    - load estimation
    """

    def __init__(
        self,
        config: CurrentSensorConfig,
        source: CurrentSource,
    ) -> None:
        self.config = config
        self.source = source

        self._last_reading: CurrentReading | None = None

    async def read(self) -> CurrentReading:
        try:
            raw = await self.source.read_current()

            value = self._calibrate(raw)

            quality = self._quality(value)

            status = self._status(
                value,
                quality,
            )

            reading = CurrentReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                current_a=round(value, 4),
                phase=self.config.phase,
                status=status,
                quality=quality,
                metadata={
                    "raw_current": raw,
                    "unit": self.config.unit,
                },
            )

            self._last_reading = reading

            return reading

        except Exception as exc:
            logger.exception(
                "Current sensor read failed sensor=%s",
                self.config.sensor_id,
            )

            reading = CurrentReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                current_a=0.0,
                phase=self.config.phase,
                status=CurrentStatus.SENSOR_FAULT,
                quality=CurrentQuality.BAD,
                metadata={
                    "error": str(exc),
                },
            )

            self._last_reading = reading

            return reading

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

    def _quality(
        self,
        value: float,
    ) -> CurrentQuality:
        if not math.isfinite(value):
            return CurrentQuality.BAD

        if value < 0:
            return CurrentQuality.BAD

        if (
            value < self.config.minimum_current
            or value > self.config.maximum_current
        ):
            return CurrentQuality.UNCERTAIN

        return CurrentQuality.GOOD

    def _status(
        self,
        value: float,
        quality: CurrentQuality,
    ) -> CurrentStatus:
        if quality == CurrentQuality.BAD:
            return CurrentStatus.SENSOR_FAULT

        if value >= self.config.critical_high:
            return CurrentStatus.OVERCURRENT

        if value <= self.config.critical_low:
            return CurrentStatus.UNDERCURRENT

        if (
            value >= self.config.warning_high
            or value <= self.config.warning_low
        ):
            return CurrentStatus.WARNING

        return CurrentStatus.NORMAL

    async def stream(
        self,
        *,
        interval: float | None = None,
    ) -> AsyncIterator[CurrentReading]:
        delay = (
            interval
            if interval is not None
            else self.config.sample_interval
        )

        if delay <= 0:
            raise ValueError(
                "Current sensor interval must be positive"
            )

        while True:
            yield await self.read()

            await asyncio.sleep(delay)

    @property
    def last_reading(
        self,
    ) -> CurrentReading | None:
        return self._last_reading


class SimulatedCurrentSource:
    """
    Simulated current source for local/demo/test use.

    Supports load variation, degradation and
    controlled overcurrent fault injection.
    """

    def __init__(
        self,
        *,
        nominal_current: float = 55.0,
        noise: float = 1.5,
        seed: int | None = None,
    ) -> None:
        self.nominal_current = nominal_current
        self.noise = max(0.0, noise)

        self.load_percent = 70.0
        self.degradation = 0.0

        self._overcurrent = False
        self._undercurrent = False
        self._disconnected = False

        self._rng = random.Random(seed)

    async def read_current(self) -> float:
        await asyncio.sleep(0)

        if self._disconnected:
            raise ConnectionError(
                "Current sensor disconnected"
            )

        load_factor = max(
            0.0,
            self.load_percent / 100.0,
        )

        value = (
            self.nominal_current
            * load_factor
        )

        value += (
            self.degradation
            * self.nominal_current
            * 0.25
        )

        value += self._rng.uniform(
            -self.noise,
            self.noise,
        )

        if self._overcurrent:
            value += (
                self.nominal_current
                * 1.3
            )

        if self._undercurrent:
            value *= 0.05

        return max(
            0.0,
            value,
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

    def inject_overcurrent(self) -> None:
        self._overcurrent = True
        self._undercurrent = False

    def inject_undercurrent(self) -> None:
        self._undercurrent = True
        self._overcurrent = False

    def clear_fault(self) -> None:
        self._overcurrent = False
        self._undercurrent = False
        self._disconnected = False

    def disconnect(self) -> None:
        self._disconnected = True


def calculate_rms(
    samples: list[float],
) -> float:
    """Calculate RMS current from waveform samples."""

    if not samples:
        raise ValueError(
            "At least one current sample is required"
        )

    valid = [
        float(value)
        for value in samples
        if math.isfinite(float(value))
    ]

    if not valid:
        raise ValueError(
            "No valid current samples"
        )

    mean_square = sum(
        value * value
        for value in valid
    ) / len(valid)

    return math.sqrt(mean_square)


def estimate_load_percent(
    current_a: float,
    rated_current_a: float,
) -> float:
    if rated_current_a <= 0:
        raise ValueError(
            "rated_current_a must be greater than zero"
        )

    return max(
        0.0,
        current_a
        / rated_current_a
        * 100.0,
    )


def estimate_three_phase_power_kw(
    *,
    current_a: float,
    voltage_v: float,
    power_factor: float = 0.88,
) -> float:
    """
    Approximate balanced three-phase real power:

        P = sqrt(3) × V × I × PF
    """

    if current_a < 0:
        raise ValueError(
            "current_a cannot be negative"
        )

    if voltage_v < 0:
        raise ValueError(
            "voltage_v cannot be negative"
        )

    pf = max(
        0.0,
        min(
            1.0,
            power_factor,
        ),
    )

    watts = (
        math.sqrt(3.0)
        * voltage_v
        * current_a
        * pf
    )

    return watts / 1000.0


def create_current_sensor(
    *,
    sensor_id: str,
    asset_id: str,
    site_id: str,
    nominal_current: float = 55.0,
    phase: str = "L1",
    seed: int | None = None,
) -> CurrentSensor:
    source = SimulatedCurrentSource(
        nominal_current=nominal_current,
        seed=seed,
    )

    config = CurrentSensorConfig(
        sensor_id=sensor_id,
        asset_id=asset_id,
        site_id=site_id,
        nominal_current=nominal_current,
        phase=phase,
    )

    return CurrentSensor(
        config=config,
        source=source,
    )