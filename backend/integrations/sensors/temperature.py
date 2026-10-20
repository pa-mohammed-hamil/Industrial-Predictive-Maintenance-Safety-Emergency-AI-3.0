# app/integrations/sensors/temperature.py

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, AsyncIterator, Protocol

logger = logging.getLogger(__name__)


class TemperatureStatus(str, Enum):
    NORMAL = "normal"
    WARNING_LOW = "warning_low"
    WARNING_HIGH = "warning_high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"
    RAPID_RISE = "rapid_rise"
    OVERHEATING = "overheating"
    SENSOR_FAULT = "sensor_fault"


class TemperatureQuality(str, Enum):
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"


@dataclass(slots=True)
class TemperatureSensorConfig:
    sensor_id: str
    asset_id: str
    site_id: str

    location: str = "machine"
    unit: str = "°C"

    sample_interval: float = 1.0

    nominal_temperature: float = 55.0

    minimum_temperature: float = -40.0
    maximum_temperature: float = 200.0

    warning_low: float = 5.0
    warning_high: float = 80.0

    critical_low: float = -10.0
    critical_high: float = 100.0

    overheating_threshold: float = 110.0

    rapid_rise_threshold_c_per_min: float = 10.0

    calibration_scale: float = 1.0
    calibration_offset: float = 0.0

    moving_average_window: int = 10


@dataclass(slots=True)
class TemperatureReading:
    sensor_id: str
    asset_id: str
    site_id: str

    temperature_c: float
    location: str

    status: TemperatureStatus
    quality: TemperatureQuality

    temperature_change: float = 0.0
    rate_c_per_min: float = 0.0

    moving_average_c: float | None = None

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
            "temperature_c": self.temperature_c,
            "location": self.location,
            "status": self.status.value,
            "quality": self.quality.value,
            "temperature_change": self.temperature_change,
            "rate_c_per_min": self.rate_c_per_min,
            "moving_average_c": self.moving_average_c,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class TemperatureSource(Protocol):
    async def read_temperature(self) -> float:
        ...


class TemperatureSensor:
    """
    Industrial temperature sensor integration.

    Supports:
    - calibrated temperature readings
    - warning/critical thresholds
    - overheating detection
    - rapid temperature-rise detection
    - moving-average smoothing
    - rate-of-change calculation
    - sensor quality checks
    - async streaming
    """

    def __init__(
        self,
        config: TemperatureSensorConfig,
        source: TemperatureSource,
    ) -> None:
        self.config = config
        self.source = source

        window_size = max(
            1,
            config.moving_average_window,
        )

        self._history: deque[float] = deque(
            maxlen=window_size
        )

        self._last_value: float | None = None
        self._last_timestamp: datetime | None = None
        self._last_reading: TemperatureReading | None = None

    async def read(self) -> TemperatureReading:
        now = datetime.now(timezone.utc)

        try:
            raw_value = (
                await self.source.read_temperature()
            )

            temperature = self._calibrate(
                raw_value
            )

            quality = self._evaluate_quality(
                temperature
            )

            change = 0.0
            rate_per_minute = 0.0

            if self._last_value is not None:
                change = (
                    temperature
                    - self._last_value
                )

            if self._last_timestamp is not None:
                elapsed_seconds = (
                    now - self._last_timestamp
                ).total_seconds()

                if elapsed_seconds > 0:
                    rate_per_minute = (
                        change
                        / elapsed_seconds
                        * 60.0
                    )

            if quality != TemperatureQuality.BAD:
                self._history.append(
                    temperature
                )

            moving_average = (
                mean(self._history)
                if self._history
                else None
            )

            status = self._evaluate_status(
                temperature=temperature,
                rate_c_per_min=rate_per_minute,
                quality=quality,
            )

            reading = TemperatureReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                temperature_c=round(
                    temperature,
                    4,
                ),
                location=self.config.location,
                status=status,
                quality=quality,
                temperature_change=round(
                    change,
                    4,
                ),
                rate_c_per_min=round(
                    rate_per_minute,
                    4,
                ),
                moving_average_c=(
                    round(
                        moving_average,
                        4,
                    )
                    if moving_average is not None
                    else None
                ),
                timestamp=now,
                metadata={
                    "unit": self.config.unit,
                    "raw_temperature": raw_value,
                    "nominal_temperature": (
                        self.config.nominal_temperature
                    ),
                },
            )

            if quality != TemperatureQuality.BAD:
                self._last_value = temperature
                self._last_timestamp = now

            self._last_reading = reading

            return reading

        except Exception as exc:
            logger.exception(
                "Temperature sensor read failed "
                "sensor=%s",
                self.config.sensor_id,
            )

            reading = TemperatureReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                temperature_c=0.0,
                location=self.config.location,
                status=(
                    TemperatureStatus.SENSOR_FAULT
                ),
                quality=TemperatureQuality.BAD,
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
        temperature: float,
    ) -> TemperatureQuality:
        if not math.isfinite(temperature):
            return TemperatureQuality.BAD

        if (
            temperature
            < self.config.minimum_temperature
            or temperature
            > self.config.maximum_temperature
        ):
            return TemperatureQuality.UNCERTAIN

        return TemperatureQuality.GOOD

    # --------------------------------------------------
    # Status
    # --------------------------------------------------

    def _evaluate_status(
        self,
        *,
        temperature: float,
        rate_c_per_min: float,
        quality: TemperatureQuality,
    ) -> TemperatureStatus:
        if quality == TemperatureQuality.BAD:
            return TemperatureStatus.SENSOR_FAULT

        if (
            temperature
            >= self.config.overheating_threshold
        ):
            return TemperatureStatus.OVERHEATING

        if (
            temperature
            >= self.config.critical_high
        ):
            return TemperatureStatus.CRITICAL_HIGH

        if (
            temperature
            <= self.config.critical_low
        ):
            return TemperatureStatus.CRITICAL_LOW

        if (
            rate_c_per_min
            >= self.config.rapid_rise_threshold_c_per_min
        ):
            return TemperatureStatus.RAPID_RISE

        if (
            temperature
            >= self.config.warning_high
        ):
            return TemperatureStatus.WARNING_HIGH

        if (
            temperature
            <= self.config.warning_low
        ):
            return TemperatureStatus.WARNING_LOW

        return TemperatureStatus.NORMAL

    # --------------------------------------------------
    # Streaming
    # --------------------------------------------------

    async def stream(
        self,
        *,
        interval: float | None = None,
    ) -> AsyncIterator[TemperatureReading]:
        delay = (
            interval
            if interval is not None
            else self.config.sample_interval
        )

        if delay <= 0:
            raise ValueError(
                "Temperature sensor interval "
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
    ) -> TemperatureReading | None:
        return self._last_reading

    @property
    def moving_average(self) -> float | None:
        if not self._history:
            return None

        return mean(self._history)

    def reset_history(self) -> None:
        self._history.clear()

        self._last_value = None
        self._last_timestamp = None
        self._last_reading = None


# ---------------------------------------------------------
# Simulated temperature source
# ---------------------------------------------------------

class SimulatedTemperatureSource:
    """
    Industrial temperature simulator.

    Supports:
    - machine load
    - ambient temperature
    - equipment degradation
    - overheating
    - thermal runaway
    - cooling failure
    - sensor disconnection
    """

    def __init__(
        self,
        *,
        ambient_temperature_c: float = 28.0,
        nominal_temperature_c: float = 55.0,
        noise: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self.ambient_temperature_c = (
            ambient_temperature_c
        )

        self.nominal_temperature_c = (
            nominal_temperature_c
        )

        self.noise = max(
            0.0,
            noise,
        )

        self.load_percent = 70.0
        self.degradation = 0.0

        self._current_temperature = (
            ambient_temperature_c
        )

        self._overheating = False
        self._thermal_runaway = False
        self._cooling_failure = False
        self._disconnected = False

        self._rng = random.Random(seed)

    async def read_temperature(self) -> float:
        await asyncio.sleep(0)

        if self._disconnected:
            raise ConnectionError(
                "Temperature sensor disconnected"
            )

        load_factor = max(
            0.0,
            self.load_percent / 100.0,
        )

        target = (
            self.ambient_temperature_c
            + (
                self.nominal_temperature_c
                - self.ambient_temperature_c
            )
            * load_factor
        )

        target += (
            self.degradation * 25.0
        )

        if self._cooling_failure:
            target += 25.0

        if self._overheating:
            target = max(
                target,
                115.0,
            )

        if self._thermal_runaway:
            target = max(
                target,
                self._current_temperature
                + 8.0,
            )

        heating_rate = (
            0.20
            if not self._thermal_runaway
            else 0.55
        )

        self._current_temperature += (
            target
            - self._current_temperature
        ) * heating_rate

        self._current_temperature += (
            self._rng.uniform(
                -self.noise,
                self.noise,
            )
        )

        return self._current_temperature

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

    def set_ambient_temperature(
        self,
        temperature_c: float,
    ) -> None:
        self.ambient_temperature_c = float(
            temperature_c
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

    def inject_overheating(self) -> None:
        self._overheating = True

    def inject_thermal_runaway(self) -> None:
        self._thermal_runaway = True

    def inject_cooling_failure(self) -> None:
        self._cooling_failure = True

    def disconnect(self) -> None:
        self._disconnected = True

    def clear_faults(self) -> None:
        self._overheating = False
        self._thermal_runaway = False
        self._cooling_failure = False
        self._disconnected = False


# ---------------------------------------------------------
# Temperature utilities
# ---------------------------------------------------------

def celsius_to_fahrenheit(
    temperature_c: float,
) -> float:
    return (
        float(temperature_c)
        * 9.0
        / 5.0
        + 32.0
    )


def fahrenheit_to_celsius(
    temperature_f: float,
) -> float:
    return (
        float(temperature_f)
        - 32.0
    ) * 5.0 / 9.0


def temperature_deviation(
    *,
    measured_temperature: float,
    nominal_temperature: float,
) -> float:
    return (
        float(measured_temperature)
        - float(nominal_temperature)
    )


def temperature_deviation_percent(
    *,
    measured_temperature: float,
    nominal_temperature: float,
) -> float:
    if nominal_temperature == 0:
        raise ValueError(
            "nominal_temperature cannot be zero"
        )

    return (
        measured_temperature
        - nominal_temperature
    ) / abs(nominal_temperature) * 100.0


def calculate_temperature_health_score(
    *,
    temperature_c: float,
    nominal_temperature_c: float,
    critical_temperature_c: float,
) -> float:
    """
    Return a simple 0-100 thermal health score.

    100 -> nominal or cooler.
    0   -> critical temperature or hotter.
    """

    if (
        critical_temperature_c
        <= nominal_temperature_c
    ):
        raise ValueError(
            "critical_temperature_c must be "
            "greater than nominal_temperature_c"
        )

    if temperature_c <= nominal_temperature_c:
        return 100.0

    ratio = (
        temperature_c
        - nominal_temperature_c
    ) / (
        critical_temperature_c
        - nominal_temperature_c
    )

    return max(
        0.0,
        min(
            100.0,
            100.0 * (1.0 - ratio),
        ),
    )


def detect_overheating(
    *,
    temperature_c: float,
    threshold_c: float,
) -> bool:
    return (
        temperature_c
        >= threshold_c
    )


def detect_rapid_temperature_rise(
    *,
    previous_temperature_c: float,
    current_temperature_c: float,
    elapsed_seconds: float,
    threshold_c_per_min: float = 10.0,
) -> bool:
    if elapsed_seconds <= 0:
        return False

    change = (
        current_temperature_c
        - previous_temperature_c
    )

    rate = (
        change
        / elapsed_seconds
        * 60.0
    )

    return (
        rate
        >= threshold_c_per_min
    )


# ---------------------------------------------------------
# Factory
# ---------------------------------------------------------

def create_temperature_sensor(
    *,
    sensor_id: str,
    asset_id: str,
    site_id: str,
    location: str = "motor",
    nominal_temperature_c: float = 55.0,
    ambient_temperature_c: float = 28.0,
    seed: int | None = None,
) -> TemperatureSensor:
    source = SimulatedTemperatureSource(
        ambient_temperature_c=(
            ambient_temperature_c
        ),
        nominal_temperature_c=(
            nominal_temperature_c
        ),
        seed=seed,
    )

    config = TemperatureSensorConfig(
        sensor_id=sensor_id,
        asset_id=asset_id,
        site_id=site_id,
        location=location,
        nominal_temperature=(
            nominal_temperature_c
        ),
    )

    return TemperatureSensor(
        config=config,
        source=source,
    )