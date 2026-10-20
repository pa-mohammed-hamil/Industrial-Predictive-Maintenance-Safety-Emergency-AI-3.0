# app/integrations/sensors/vibration.py

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
from typing import Any, AsyncIterator, Protocol, Sequence

logger = logging.getLogger(__name__)


class VibrationStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HIGH_VIBRATION = "high_vibration"
    SHOCK = "shock"
    RAPID_INCREASE = "rapid_increase"
    SENSOR_FAULT = "sensor_fault"


class VibrationQuality(str, Enum):
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"


@dataclass(slots=True)
class VibrationSample:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class VibrationSensorConfig:
    sensor_id: str
    asset_id: str
    site_id: str

    location: str = "bearing"
    unit: str = "mm/s"

    sample_interval: float = 1.0

    nominal_rms: float = 2.0

    warning_rms: float = 4.5
    critical_rms: float = 7.1

    shock_threshold: float = 12.0
    maximum_rms: float = 50.0

    rapid_increase_threshold: float = 2.0

    calibration_scale: float = 1.0
    calibration_offset: float = 0.0

    moving_average_window: int = 10


@dataclass(slots=True)
class VibrationReading:
    sensor_id: str
    asset_id: str
    site_id: str

    x: float
    y: float
    z: float

    rms: float
    peak: float
    vector_magnitude: float

    status: VibrationStatus
    quality: VibrationQuality

    rms_change: float = 0.0
    moving_average_rms: float | None = None

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "asset_id": self.asset_id,
            "site_id": self.site_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rms": self.rms,
            "peak": self.peak,
            "vector_magnitude": self.vector_magnitude,
            "status": self.status.value,
            "quality": self.quality.value,
            "rms_change": self.rms_change,
            "moving_average_rms": self.moving_average_rms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class VibrationSource(Protocol):
    async def read_vibration(self) -> VibrationSample:
        ...


class VibrationSensor:
    """
    Industrial vibration sensor integration.

    Handles:
    - X/Y/Z vibration
    - RMS calculation
    - peak vibration
    - vector magnitude
    - moving-average RMS
    - rapid vibration increase
    - shock detection
    - warning/critical classification
    - async telemetry streaming
    """

    def __init__(
        self,
        config: VibrationSensorConfig,
        source: VibrationSource,
    ) -> None:
        self.config = config
        self.source = source

        self._history: deque[float] = deque(
            maxlen=max(1, config.moving_average_window)
        )

        self._last_rms: float | None = None
        self._last_reading: VibrationReading | None = None

    async def read(self) -> VibrationReading:
        now = datetime.now(timezone.utc)

        try:
            sample = await self.source.read_vibration()

            x = self._calibrate(sample.x)
            y = self._calibrate(sample.y)
            z = self._calibrate(sample.z)

            quality = self._evaluate_quality(x, y, z)

            rms = calculate_axis_rms(x, y, z)

            vector_magnitude = calculate_vector_magnitude(
                x=x,
                y=y,
                z=z,
            )

            peak = max(
                abs(x),
                abs(y),
                abs(z),
            )

            rms_change = 0.0

            if self._last_rms is not None:
                rms_change = rms - self._last_rms

            if quality != VibrationQuality.BAD:
                self._history.append(rms)

            moving_average = (
                mean(self._history)
                if self._history
                else None
            )

            status = self._evaluate_status(
                rms=rms,
                peak=peak,
                rms_change=rms_change,
                quality=quality,
            )

            reading = VibrationReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                x=round(x, 5),
                y=round(y, 5),
                z=round(z, 5),
                rms=round(rms, 5),
                peak=round(peak, 5),
                vector_magnitude=round(
                    vector_magnitude,
                    5,
                ),
                status=status,
                quality=quality,
                rms_change=round(rms_change, 5),
                moving_average_rms=(
                    round(moving_average, 5)
                    if moving_average is not None
                    else None
                ),
                timestamp=now,
                metadata={
                    "location": self.config.location,
                    "unit": self.config.unit,
                    "nominal_rms": self.config.nominal_rms,
                },
            )

            if quality != VibrationQuality.BAD:
                self._last_rms = rms

            self._last_reading = reading

            return reading

        except Exception as exc:
            logger.exception(
                "Vibration sensor read failed sensor=%s",
                self.config.sensor_id,
            )

            reading = VibrationReading(
                sensor_id=self.config.sensor_id,
                asset_id=self.config.asset_id,
                site_id=self.config.site_id,
                x=0.0,
                y=0.0,
                z=0.0,
                rms=0.0,
                peak=0.0,
                vector_magnitude=0.0,
                status=VibrationStatus.SENSOR_FAULT,
                quality=VibrationQuality.BAD,
                timestamp=now,
                metadata={
                    "error": str(exc),
                    "location": self.config.location,
                },
            )

            self._last_reading = reading

            return reading

    def _calibrate(self, value: float) -> float:
        value = float(value)

        return (
            value * self.config.calibration_scale
            + self.config.calibration_offset
        )

    def _evaluate_quality(
        self,
        x: float,
        y: float,
        z: float,
    ) -> VibrationQuality:
        values = (x, y, z)

        if not all(math.isfinite(value) for value in values):
            return VibrationQuality.BAD

        if any(
            abs(value) > self.config.maximum_rms * 3.0
            for value in values
        ):
            return VibrationQuality.UNCERTAIN

        return VibrationQuality.GOOD

    def _evaluate_status(
        self,
        *,
        rms: float,
        peak: float,
        rms_change: float,
        quality: VibrationQuality,
    ) -> VibrationStatus:
        if quality == VibrationQuality.BAD:
            return VibrationStatus.SENSOR_FAULT

        if peak >= self.config.shock_threshold:
            return VibrationStatus.SHOCK

        if rms >= self.config.critical_rms:
            return VibrationStatus.CRITICAL

        if (
            self._last_rms is not None
            and rms_change
            >= self.config.rapid_increase_threshold
        ):
            return VibrationStatus.RAPID_INCREASE

        if rms >= self.config.warning_rms:
            return VibrationStatus.HIGH_VIBRATION

        return VibrationStatus.NORMAL

    async def stream(
        self,
        *,
        interval: float | None = None,
    ) -> AsyncIterator[VibrationReading]:
        delay = (
            interval
            if interval is not None
            else self.config.sample_interval
        )

        if delay <= 0:
            raise ValueError(
                "Vibration interval must be positive"
            )

        while True:
            yield await self.read()
            await asyncio.sleep(delay)

    @property
    def last_reading(
        self,
    ) -> VibrationReading | None:
        return self._last_reading

    @property
    def moving_average_rms(
        self,
    ) -> float | None:
        if not self._history:
            return None

        return mean(self._history)

    def reset_history(self) -> None:
        self._history.clear()
        self._last_rms = None
        self._last_reading = None


class SimulatedVibrationSource:
    """
    Simulated rotating-equipment vibration source.

    Fault injection includes bearing wear,
    imbalance, misalignment, looseness and shock.
    """

    def __init__(
        self,
        *,
        nominal_rms: float = 2.0,
        noise: float = 0.15,
        seed: int | None = None,
    ) -> None:
        self.nominal_rms = nominal_rms
        self.noise = max(0.0, noise)

        self.load_percent = 70.0
        self.speed_rpm = 1500.0
        self.degradation = 0.0

        self._bearing_wear = False
        self._imbalance = False
        self._misalignment = False
        self._looseness = False
        self._shock = False
        self._disconnected = False

        self._rng = random.Random(seed)

    async def read_vibration(self) -> VibrationSample:
        await asyncio.sleep(0)

        if self._disconnected:
            raise ConnectionError(
                "Vibration sensor disconnected"
            )

        load_factor = max(
            0.0,
            self.load_percent / 100.0,
        )

        speed_factor = max(
            0.1,
            self.speed_rpm / 1500.0,
        )

        base = (
            self.nominal_rms
            * (0.5 + 0.5 * load_factor)
            * speed_factor
        )

        base += (
            self.degradation
            * self.nominal_rms
            * 1.5
        )

        x = base + self._noise()
        y = base * 0.90 + self._noise()
        z = base * 0.75 + self._noise()

        if self._bearing_wear:
            x += 2.0 + self.degradation * 3.0
            y += 1.5 + self.degradation * 2.0
            z += 1.0

        if self._imbalance:
            x += 3.5
            y += 1.0

        if self._misalignment:
            x += 2.0
            y += 2.5
            z += 1.5

        if self._looseness:
            x += self._rng.uniform(2.0, 4.0)
            y += self._rng.uniform(2.0, 4.0)
            z += self._rng.uniform(1.0, 3.0)

        if self._shock:
            shock = self._rng.uniform(
                12.0,
                18.0,
            )

            x += shock
            y += shock * 0.7
            z += shock * 0.5

            # One-shot event.
            self._shock = False

        return VibrationSample(
            x=max(0.0, x),
            y=max(0.0, y),
            z=max(0.0, z),
        )

    def _noise(self) -> float:
        return self._rng.uniform(
            -self.noise,
            self.noise,
        )

    def set_load(self, percent: float) -> None:
        self.load_percent = max(
            0.0,
            min(150.0, float(percent)),
        )

    def set_speed(self, rpm: float) -> None:
        self.speed_rpm = max(
            0.0,
            float(rpm),
        )

    def set_degradation(self, value: float) -> None:
        self.degradation = max(
            0.0,
            min(1.0, float(value)),
        )

    def inject_bearing_wear(self) -> None:
        self._bearing_wear = True

    def inject_imbalance(self) -> None:
        self._imbalance = True

    def inject_misalignment(self) -> None:
        self._misalignment = True

    def inject_looseness(self) -> None:
        self._looseness = True

    def inject_shock(self) -> None:
        self._shock = True

    def disconnect(self) -> None:
        self._disconnected = True

    def clear_faults(self) -> None:
        self._bearing_wear = False
        self._imbalance = False
        self._misalignment = False
        self._looseness = False
        self._shock = False
        self._disconnected = False


def calculate_axis_rms(
    x: float,
    y: float,
    z: float,
) -> float:
    return math.sqrt(
        (
            float(x) ** 2
            + float(y) ** 2
            + float(z) ** 2
        )
        / 3.0
    )


def calculate_vector_magnitude(
    *,
    x: float,
    y: float,
    z: float,
) -> float:
    return math.sqrt(
        x * x
        + y * y
        + z * z
    )


def calculate_rms(
    samples: Sequence[float],
) -> float:
    if not samples:
        raise ValueError(
            "At least one vibration sample is required"
        )

    values = [
        float(value)
        for value in samples
    ]

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Vibration samples must be finite"
        )

    return math.sqrt(
        sum(value * value for value in values)
        / len(values)
    )


def calculate_peak(
    samples: Sequence[float],
) -> float:
    if not samples:
        raise ValueError(
            "At least one sample is required"
        )

    return max(
        abs(float(value))
        for value in samples
    )


def calculate_crest_factor(
    samples: Sequence[float],
) -> float:
    rms = calculate_rms(samples)

    if rms == 0:
        return 0.0

    return calculate_peak(samples) / rms


def calculate_vibration_health_score(
    *,
    rms: float,
    nominal_rms: float,
    critical_rms: float,
) -> float:
    if critical_rms <= nominal_rms:
        raise ValueError(
            "critical_rms must be greater "
            "than nominal_rms"
        )

    if rms <= nominal_rms:
        return 100.0

    ratio = (
        rms - nominal_rms
    ) / (
        critical_rms - nominal_rms
    )

    return max(
        0.0,
        min(
            100.0,
            100.0 * (1.0 - ratio),
        ),
    )


def detect_vibration_anomaly(
    *,
    rms: float,
    baseline_rms: float,
    tolerance_percent: float = 50.0,
) -> bool:
    if baseline_rms <= 0:
        return False

    deviation = (
        abs(rms - baseline_rms)
        / baseline_rms
        * 100.0
    )

    return deviation >= tolerance_percent


def create_vibration_sensor(
    *,
    sensor_id: str,
    asset_id: str,
    site_id: str,
    location: str = "bearing",
    nominal_rms: float = 2.0,
    seed: int | None = None,
) -> VibrationSensor:
    source = SimulatedVibrationSource(
        nominal_rms=nominal_rms,
        seed=seed,
    )

    config = VibrationSensorConfig(
        sensor_id=sensor_id,
        asset_id=asset_id,
        site_id=site_id,
        location=location,
        nominal_rms=nominal_rms,
    )

    return VibrationSensor(
        config=config,
        source=source,
    )