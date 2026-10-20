# app/ml/anomaly_detection/sensor_anomaly.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence
from uuid import uuid4


# ============================================================
# Enums
# ============================================================


class SensorAnomalyType(str, Enum):
    OUT_OF_RANGE = "out_of_range"
    SPIKE = "spike"
    DROP = "drop"
    STUCK = "stuck"
    FLATLINE = "flatline"
    DRIFT = "drift"
    EXCESSIVE_NOISE = "excessive_noise"
    RAPID_CHANGE = "rapid_change"
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    TIMESTAMP_GAP = "timestamp_gap"
    OUT_OF_ORDER = "out_of_order"
    INVALID_VALUE = "invalid_value"
    LOW_CONFIDENCE = "low_confidence"
    SENSOR_DISAGREEMENT = "sensor_disagreement"
    CALIBRATION_SUSPECTED = "calibration_suspected"
    UNKNOWN = "unknown"


class SensorSeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SensorHealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    FAILED = "failed"
    UNKNOWN = "unknown"


class SensorQuality(str, Enum):
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    MISSING = "missing"


# ============================================================
# Data Models
# ============================================================


@dataclass(slots=True)
class SensorReading:
    sensor_id: str
    timestamp: datetime

    value: float | None

    asset_id: str | None = None
    sensor_type: str | None = None
    unit: str | None = None
    site_id: str | None = None
    zone_id: str | None = None

    confidence: float = 1.0
    quality: SensorQuality = SensorQuality.GOOD

    sequence: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(
                tzinfo=timezone.utc
            )


@dataclass(slots=True)
class SensorLimits:
    """
    Physical and operational limits for a sensor.

    These should normally be loaded from the asset/sensor registry.
    """

    physical_min: float | None = None
    physical_max: float | None = None

    operating_min: float | None = None
    operating_max: float | None = None

    maximum_rate_of_change: float | None = None

    expected_interval_seconds: float | None = None
    maximum_gap_seconds: float | None = None

    minimum_confidence: float = 0.60


@dataclass(slots=True)
class SensorPolicy:
    min_history: int = 10
    max_history: int = 500

    spike_z_score: float = 4.0
    drop_z_score: float = 4.0

    stuck_window: int = 8
    stuck_tolerance: float = 1e-6

    flatline_window: int = 12
    flatline_std_threshold: float = 1e-5

    noise_window: int = 20
    noise_std_ratio: float = 0.20

    drift_window: int = 20
    drift_z_score: float = 2.5

    stale_multiplier: float = 3.0

    medium_score: float = 0.40
    high_score: float = 0.70
    critical_score: float = 0.90


@dataclass(slots=True)
class SensorEvidence:
    metric: str
    value: float | str | None

    expected: float | str | None = None

    score: float = 0.0
    description: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SensorAnomaly:
    anomaly_id: str

    anomaly_type: SensorAnomalyType
    severity: SensorSeverity

    score: float
    confidence: float

    timestamp: datetime

    sensor_id: str
    asset_id: str | None

    evidence: list[SensorEvidence] = field(default_factory=list)

    invalidate_reading: bool = False
    requires_investigation: bool = False
    requires_calibration: bool = False
    requires_replacement_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SensorAssessment:
    assessment_id: str

    timestamp: datetime

    sensor_id: str
    asset_id: str | None

    health_state: SensorHealthState
    severity: SensorSeverity

    anomaly_score: float
    confidence: float

    anomalies: list[SensorAnomaly]
    evidence: list[SensorEvidence]

    reading_valid: bool
    usable_for_ml: bool
    usable_for_safety: bool

    requires_investigation: bool = False
    requires_calibration: bool = False
    requires_replacement_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        return self.severity != SensorSeverity.NORMAL

    @property
    def is_failed(self) -> bool:
        return self.health_state == SensorHealthState.FAILED


# ============================================================
# Sensor History
# ============================================================


class SensorHistory:
    def __init__(self, max_history: int = 500) -> None:
        self._readings: deque[SensorReading] = deque(
            maxlen=max_history
        )

    def add(self, reading: SensorReading) -> None:
        self._readings.append(reading)

    @property
    def readings(self) -> list[SensorReading]:
        return list(self._readings)

    @property
    def size(self) -> int:
        return len(self._readings)

    def values(self) -> list[float]:
        return [
            float(reading.value)
            for reading in self._readings
            if _valid_number(reading.value)
        ]

    def recent_values(self, window: int) -> list[float]:
        readings = list(self._readings)[-window:]

        return [
            float(reading.value)
            for reading in readings
            if _valid_number(reading.value)
        ]

    @property
    def latest(self) -> SensorReading | None:
        if not self._readings:
            return None
        return self._readings[-1]


# ============================================================
# Detector
# ============================================================


class SensorAnomalyDetector:
    """
    Deterministic sensor-health anomaly detector.

    Detects problems in the measurement system itself rather than
    assuming every unusual measurement is a machine anomaly.

    Example:

        temperature jumps from 65 C -> 900 C

    This module determines whether that could be:
        - impossible sensor value
        - spike
        - corrupted telemetry
        - failed sensor

    before downstream predictive-maintenance models consume it.
    """

    _SEVERITY_RANK = {
        SensorSeverity.NORMAL: 0,
        SensorSeverity.LOW: 1,
        SensorSeverity.MEDIUM: 2,
        SensorSeverity.HIGH: 3,
        SensorSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        limits: SensorLimits | None = None,
        policy: SensorPolicy | None = None,
    ) -> None:

        self.limits = limits or SensorLimits()
        self.policy = policy or SensorPolicy()

        self._histories: dict[str, SensorHistory] = {}

    # ========================================================
    # Public API
    # ========================================================

    def assess(
        self,
        reading: SensorReading,
        *,
        update_history: bool = True,
    ) -> SensorAssessment:

        history = self._history(reading.sensor_id)

        anomalies: list[SensorAnomaly] = []

        anomalies.extend(
            self._check_validity(reading)
        )

        anomalies.extend(
            self._check_timestamp(reading, history)
        )

        anomalies.extend(
            self._check_range(reading)
        )

        anomalies.extend(
            self._check_rate_of_change(
                reading,
                history,
            )
        )

        if history.size >= self.policy.min_history:
            anomalies.extend(
                self._check_statistical_anomaly(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_stuck_sensor(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_flatline(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_noise(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_drift(
                    reading,
                    history,
                )
            )

        anomalies = self._deduplicate(anomalies)

        severity = self._overall_severity(anomalies)
        anomaly_score = self._overall_score(anomalies)

        invalid = any(
            anomaly.invalidate_reading
            for anomaly in anomalies
        )

        health_state = self._health_state(
            severity,
            invalid,
        )

        requires_calibration = any(
            anomaly.requires_calibration
            for anomaly in anomalies
        )

        requires_replacement = any(
            anomaly.requires_replacement_review
            for anomaly in anomalies
        )

        requires_investigation = bool(anomalies)

        evidence = [
            evidence
            for anomaly in anomalies
            for evidence in anomaly.evidence
        ]

        reading_valid = (
            not invalid
            and _valid_number(reading.value)
        )

        # ML should be stricter than basic ingestion.
        usable_for_ml = (
            reading_valid
            and severity
            not in {
                SensorSeverity.HIGH,
                SensorSeverity.CRITICAL,
            }
        )

        # Never silently trust a known-bad sensor in safety logic.
        usable_for_safety = (
            reading_valid
            and health_state
            in {
                SensorHealthState.HEALTHY,
                SensorHealthState.DEGRADED,
            }
        )

        assessment = SensorAssessment(
            assessment_id=str(uuid4()),
            timestamp=reading.timestamp,
            sensor_id=reading.sensor_id,
            asset_id=reading.asset_id,
            health_state=health_state,
            severity=severity,
            anomaly_score=anomaly_score,
            confidence=reading.confidence,
            anomalies=anomalies,
            evidence=evidence,
            reading_valid=reading_valid,
            usable_for_ml=usable_for_ml,
            usable_for_safety=usable_for_safety,
            requires_investigation=requires_investigation,
            requires_calibration=requires_calibration,
            requires_replacement_review=requires_replacement,
            metadata={
                "history_size": history.size,
                "sensor_type": reading.sensor_type,
                "unit": reading.unit,
                "site_id": reading.site_id,
                "zone_id": reading.zone_id,
            },
        )

        if update_history:
            self._update_history(
                reading,
                assessment,
            )

        return assessment

    def detect(
        self,
        reading: SensorReading,
    ) -> list[SensorAnomaly]:
        return self.assess(reading).anomalies

    def assess_many(
        self,
        readings: Iterable[SensorReading],
    ) -> list[SensorAssessment]:

        ordered = sorted(
            readings,
            key=lambda item: item.timestamp,
        )

        return [
            self.assess(reading)
            for reading in ordered
        ]

    # ========================================================
    # Basic validity
    # ========================================================

    def _check_validity(
        self,
        reading: SensorReading,
    ) -> list[SensorAnomaly]:

        if reading.value is None:
            return [
                self._anomaly(
                    reading,
                    SensorAnomalyType.MISSING_DATA,
                    SensorSeverity.HIGH,
                    0.90,
                    [
                        SensorEvidence(
                            metric="value",
                            value=None,
                            score=0.90,
                            description=(
                                "Sensor reading is missing"
                            ),
                        )
                    ],
                    invalidate=True,
                )
            ]

        if not _valid_number(reading.value):
            return [
                self._anomaly(
                    reading,
                    SensorAnomalyType.INVALID_VALUE,
                    SensorSeverity.CRITICAL,
                    1.0,
                    [
                        SensorEvidence(
                            metric="value",
                            value=str(reading.value),
                            score=1.0,
                            description=(
                                "Sensor returned non-finite value"
                            ),
                        )
                    ],
                    invalidate=True,
                    replacement=True,
                )
            ]

        if (
            reading.confidence
            < self.limits.minimum_confidence
        ):
            return [
                self._anomaly(
                    reading,
                    SensorAnomalyType.LOW_CONFIDENCE,
                    SensorSeverity.MEDIUM,
                    0.60,
                    [
                        SensorEvidence(
                            metric="confidence",
                            value=reading.confidence,
                            expected=(
                                self.limits.minimum_confidence
                            ),
                            score=0.60,
                            description=(
                                "Sensor confidence below "
                                "minimum accepted level"
                            ),
                        )
                    ],
                )
            ]

        return []

    # ========================================================
    # Physical / operational range
    # ========================================================

    def _check_range(
        self,
        reading: SensorReading,
    ) -> list[SensorAnomaly]:

        if not _valid_number(reading.value):
            return []

        value = float(reading.value)
        limits = self.limits

        if (
            limits.physical_min is not None
            and value < limits.physical_min
        ):
            return [
                self._out_of_range(
                    reading,
                    value,
                    limits.physical_min,
                    "below physical sensor minimum",
                    critical=True,
                )
            ]

        if (
            limits.physical_max is not None
            and value > limits.physical_max
        ):
            return [
                self._out_of_range(
                    reading,
                    value,
                    limits.physical_max,
                    "above physical sensor maximum",
                    critical=True,
                )
            ]

        if (
            limits.operating_min is not None
            and value < limits.operating_min
        ):
            return [
                self._out_of_range(
                    reading,
                    value,
                    limits.operating_min,
                    "below expected operating range",
                    critical=False,
                )
            ]

        if (
            limits.operating_max is not None
            and value > limits.operating_max
        ):
            return [
                self._out_of_range(
                    reading,
                    value,
                    limits.operating_max,
                    "above expected operating range",
                    critical=False,
                )
            ]

        return []

    def _out_of_range(
        self,
        reading: SensorReading,
        value: float,
        threshold: float,
        description: str,
        *,
        critical: bool,
    ) -> SensorAnomaly:

        return self._anomaly(
            reading,
            SensorAnomalyType.OUT_OF_RANGE,
            (
                SensorSeverity.CRITICAL
                if critical
                else SensorSeverity.HIGH
            ),
            1.0 if critical else 0.80,
            [
                SensorEvidence(
                    metric="range",
                    value=value,
                    expected=threshold,
                    score=1.0 if critical else 0.80,
                    description=description,
                )
            ],
            invalidate=critical,
            calibration=True,
        )

    # ========================================================
    # Timestamp / telemetry integrity
    # ========================================================

    def _check_timestamp(
        self,
        reading: SensorReading,
        history: SensorHistory,
    ) -> list[SensorAnomaly]:

        previous = history.latest

        if previous is None:
            return []

        delta = (
            reading.timestamp - previous.timestamp
        ).total_seconds()

        if delta < 0:
            return [
                self._anomaly(
                    reading,
                    SensorAnomalyType.OUT_OF_ORDER,
                    SensorSeverity.MEDIUM,
                    0.65,
                    [
                        SensorEvidence(
                            metric="timestamp_delta",
                            value=delta,
                            expected=">= 0 seconds",
                            score=0.65,
                            description=(
                                "Out-of-order sensor reading"
                            ),
                        )
                    ],
                    invalidate=True,
                )
            ]

        maximum_gap = self.limits.maximum_gap_seconds

        if maximum_gap is None:
            expected = (
                self.limits.expected_interval_seconds
            )

            if expected is not None:
                maximum_gap = (
                    expected
                    * self.policy.stale_multiplier
                )

        if (
            maximum_gap is not None
            and delta > maximum_gap
        ):
            score = _clamp(
                delta / max(
                    maximum_gap * 3.0,
                    1e-9,
                )
            )

            return [
                self._anomaly(
                    reading,
                    SensorAnomalyType.TIMESTAMP_GAP,
                    SensorSeverity.MEDIUM,
                    max(0.50, score),
                    [
                        SensorEvidence(
                            metric="timestamp_gap_seconds",
                            value=delta,
                            expected=maximum_gap,
                            score=score,
                            description=(
                                "Unexpected telemetry gap"
                            ),
                        )
                    ],
                )
            ]

        return []

    # ========================================================
    # Rate of change
    # ========================================================

    def _check_rate_of_change(
        self,
        reading: SensorReading,
        history: SensorHistory,
    ) -> list[SensorAnomaly]:

        limit = self.limits.maximum_rate_of_change
        previous = history.latest

        if (
            limit is None
            or previous is None
            or not _valid_number(previous.value)
            or not _valid_number(reading.value)
        ):
            return []

        delta_seconds = (
            reading.timestamp - previous.timestamp
        ).total_seconds()

        if delta_seconds <= 0:
            return []

        change = abs(
            float(reading.value)
            - float(previous.value)
        )

        rate = change / delta_seconds

        if rate <= limit:
            return []

        ratio = rate / max(limit, 1e-9)

        score = _clamp(
            0.50 + ((ratio - 1.0) * 0.25)
        )

        severity = (
            SensorSeverity.HIGH
            if ratio >= 2.0
            else SensorSeverity.MEDIUM
        )

        return [
            self._anomaly(
                reading,
                SensorAnomalyType.RAPID_CHANGE,
                severity,
                score,
                [
                    SensorEvidence(
                        metric="rate_of_change",
                        value=rate,
                        expected=limit,
                        score=score,
                        description=(
                            "Sensor changed faster than "
                            "configured physical limit"
                        ),
                    )
                ],
            )
        ]

    # ========================================================
    # Statistical spike/drop
    # ========================================================

    def _check_statistical_anomaly(
        self,
        reading: SensorReading,
        history: SensorHistory,
    ) -> list[SensorAnomaly]:

        if not _valid_number(reading.value):
            return []

        values = history.values()

        if len(values) < self.policy.min_history:
            return []

        mu = mean(values)
        sigma = pstdev(values)

        if sigma <= 1e-12:
            return []

        value = float(reading.value)

        z = (value - mu) / sigma

        absolute_z = abs(z)

        threshold = (
            self.policy.spike_z_score
            if z >= 0
            else self.policy.drop_z_score
        )

        if absolute_z < threshold:
            return []

        anomaly_type = (
            SensorAnomalyType.SPIKE
            if z > 0
            else SensorAnomalyType.DROP
        )

        score = _clamp(
            absolute_z / (threshold * 2.0)
        )

        severity = (
            SensorSeverity.HIGH
            if absolute_z >= threshold * 1.5
            else SensorSeverity.MEDIUM
        )

        return [
            self._anomaly(
                reading,
                anomaly_type,
                severity,
                max(0.60, score),
                [
                    SensorEvidence(
                        metric="z_score",
                        value=z,
                        expected=f"|z| < {threshold}",
                        score=score,
                        description=(
                            "Measurement differs strongly "
                            "from recent sensor baseline"
                        ),
                        metadata={
                            "mean": mu,
                            "std": sigma,
                        },
                    )
                ],
            )
        ]

    # ========================================================
    # Stuck sensor
    # ========================================================

    def _check_stuck_sensor(
        self,
        reading: SensorReading,
        history: SensorHistory,
    ) -> list[SensorAnomaly]:

        if not _valid_number(reading.value):
            return []

        previous = history.recent_values(
            self.policy.stuck_window - 1
        )

        values = previous + [
            float(reading.value)
        ]

        if len(values) < self.policy.stuck_window:
            return []

        spread = max(values) - min(values)

        if spread > self.policy.stuck_tolerance:
            return []

        return [
            self._anomaly(
                reading,
                SensorAnomalyType.STUCK,
                SensorSeverity.HIGH,
                0.85,
                [
                    SensorEvidence(
                        metric="window_spread",
                        value=spread,
                        expected=(
                            f"> {self.policy.stuck_tolerance}"
                        ),
                        score=0.85,
                        description=(
                            "Sensor value appears stuck "
                            "across multiple readings"
                        ),
                    )
                ],
                calibration=True,
                replacement=True,
            )
        ]

    # ========================================================
    # Flatline
    # ========================================================

    def _check_flatline(
        self,
        reading: SensorReading,
        history: SensorHistory,
    ) -> list[SensorAnomaly]:

        if not _valid_number(reading.value):
            return []

        previous = history.recent_values(
            self.policy.flatline_window - 1
        )

        values = previous + [
            float(reading.value)
        ]

        if len(values) < self.policy.flatline_window:
            return []

        std = pstdev(values)

        if std > self.policy.flatline_std_threshold:
            return []

        return [
            self._anomaly(
                reading,
                SensorAnomalyType.FLATLINE,
                SensorSeverity.MEDIUM,
                0.70,
                [
                    SensorEvidence(
                        metric="window_std",
                        value=std,
                        expected=(
                            f"> "
                            f"{self.policy.flatline_std_threshold}"
                        ),
                        score=0.70,
                        description=(
                            "Sensor variance is abnormally low"
                        ),
                    )
                ],
                calibration=True,
            )
        ]

    # ========================================================
    # Excessive noise
    # ========================================================

    def _check_noise(
        self,
        reading: SensorReading,
        history: SensorHistory,
    ) -> list[SensorAnomaly]:

        if not _valid_number(reading.value):
            return []

        values = history.recent_values(
            self.policy.noise_window - 1
        )

        values.append(float(reading.value))

        if len(values) < self.policy.noise_window:
            return []

        mu = mean(values)
        sigma = pstdev(values)

        denominator = max(
            abs(mu),
            1e-6,
        )

        coefficient_of_variation = (
            sigma / denominator
        )

        if (
            coefficient_of_variation
            <= self.policy.noise_std_ratio
        ):
            return []

        score = _clamp(
            coefficient_of_variation
            / max(
                self.policy.noise_std_ratio * 2.0,
                1e-9,
            )
        )

        return [
            self._anomaly(
                reading,
                SensorAnomalyType.EXCESSIVE_NOISE,
                SensorSeverity.MEDIUM,
                max(0.50, score),
                [
                    SensorEvidence(
                        metric="coefficient_of_variation",
                        value=coefficient_of_variation,
                        expected=(
                            self.policy.noise_std_ratio
                        ),
                        score=score,
                        description=(
                            "Sensor signal contains "
                            "excessive short-term variation"
                        ),
                    )
                ],
                calibration=True,
            )
        ]

    # ========================================================
    # Drift detection
    # ========================================================

    def _check_drift(
        self,
        reading: SensorReading,
        history: SensorHistory,
    ) -> list[SensorAnomaly]:

        if not _valid_number(reading.value):
            return []

        values = history.values()

        window = self.policy.drift_window

        if len(values) < window * 2:
            return []

        reference = values[:-window]
        recent = values[-window + 1 :] + [
            float(reading.value)
        ]

        if len(reference) < self.policy.min_history:
            return []

        reference_mean = mean(reference)
        reference_std = pstdev(reference)

        recent_mean = mean(recent)

        if reference_std <= 1e-12:
            return []

        shift_z = abs(
            recent_mean - reference_mean
        ) / reference_std

        if shift_z < self.policy.drift_z_score:
            return []

        score = _clamp(
            shift_z
            / (self.policy.drift_z_score * 2.0)
        )

        severity = (
            SensorSeverity.HIGH
            if shift_z
            >= self.policy.drift_z_score * 1.5
            else SensorSeverity.MEDIUM
        )

        return [
            self._anomaly(
                reading,
                SensorAnomalyType.DRIFT,
                severity,
                max(0.60, score),
                [
                    SensorEvidence(
                        metric="mean_shift_z_score",
                        value=shift_z,
                        expected=(
                            self.policy.drift_z_score
                        ),
                        score=score,
                        description=(
                            "Sensor baseline appears to "
                            "be drifting over time"
                        ),
                        metadata={
                            "reference_mean":
                                reference_mean,
                            "recent_mean":
                                recent_mean,
                        },
                    )
                ],
                calibration=True,
            )
        ]

    # ========================================================
    # Cross-sensor disagreement
    # ========================================================

    def compare_redundant_sensors(
        self,
        readings: Sequence[SensorReading],
        *,
        absolute_tolerance: float | None = None,
        relative_tolerance: float = 0.10,
    ) -> list[SensorAnomaly]:
        """
        Compare redundant sensors measuring the same process value.

        Example:
            temperature_sensor_A
            temperature_sensor_B
            temperature_sensor_C
        """

        valid = [
            reading
            for reading in readings
            if _valid_number(reading.value)
        ]

        if len(valid) < 2:
            return []

        values = [
            float(reading.value)
            for reading in valid
        ]

        consensus = _median(values)

        anomalies: list[SensorAnomaly] = []

        for reading in valid:
            value = float(reading.value)

            difference = abs(
                value - consensus
            )

            if absolute_tolerance is not None:
                anomalous = (
                    difference > absolute_tolerance
                )
                threshold = absolute_tolerance
            else:
                denominator = max(
                    abs(consensus),
                    1e-6,
                )

                relative_difference = (
                    difference / denominator
                )

                anomalous = (
                    relative_difference
                    > relative_tolerance
                )

                threshold = relative_tolerance

            if not anomalous:
                continue

            score = _clamp(
                difference
                / max(
                    abs(consensus),
                    threshold,
                    1e-6,
                )
            )

            anomalies.append(
                self._anomaly(
                    reading,
                    SensorAnomalyType.SENSOR_DISAGREEMENT,
                    SensorSeverity.HIGH,
                    max(0.70, score),
                    [
                        SensorEvidence(
                            metric="sensor_disagreement",
                            value=value,
                            expected=consensus,
                            score=score,
                            description=(
                                "Sensor disagrees with "
                                "redundant sensor consensus"
                            ),
                        )
                    ],
                    calibration=True,
                )
            )

        return anomalies

    # ========================================================
    # Helpers
    # ========================================================

    def _history(
        self,
        sensor_id: str,
    ) -> SensorHistory:

        history = self._histories.get(sensor_id)

        if history is None:
            history = SensorHistory(
                max_history=self.policy.max_history
            )

            self._histories[sensor_id] = history

        return history

    def _update_history(
        self,
        reading: SensorReading,
        assessment: SensorAssessment,
    ) -> None:
        """
        Invalid/non-finite readings are not added to the statistical
        baseline. Suspicious but finite readings remain in history so
        temporal fault patterns can still be observed.

        Production systems may keep separate raw and clean histories.
        """

        if not _valid_number(reading.value):
            return

        if not assessment.reading_valid:
            return

        self._history(
            reading.sensor_id
        ).add(reading)

    def _anomaly(
        self,
        reading: SensorReading,
        anomaly_type: SensorAnomalyType,
        severity: SensorSeverity,
        score: float,
        evidence: list[SensorEvidence],
        *,
        invalidate: bool = False,
        calibration: bool = False,
        replacement: bool = False,
    ) -> SensorAnomaly:

        return SensorAnomaly(
            anomaly_id=str(uuid4()),
            anomaly_type=anomaly_type,
            severity=severity,
            score=_clamp(score),
            confidence=reading.confidence,
            timestamp=reading.timestamp,
            sensor_id=reading.sensor_id,
            asset_id=reading.asset_id,
            evidence=evidence,
            invalidate_reading=invalidate,
            requires_investigation=True,
            requires_calibration=calibration,
            requires_replacement_review=replacement,
            metadata={
                "sensor_type": reading.sensor_type,
                "unit": reading.unit,
                "site_id": reading.site_id,
                "zone_id": reading.zone_id,
            },
        )

    def _overall_severity(
        self,
        anomalies: Sequence[SensorAnomaly],
    ) -> SensorSeverity:

        if not anomalies:
            return SensorSeverity.NORMAL

        return max(
            (
                anomaly.severity
                for anomaly in anomalies
            ),
            key=lambda severity:
                self._SEVERITY_RANK[severity],
        )

    @staticmethod
    def _overall_score(
        anomalies: Sequence[SensorAnomaly],
    ) -> float:

        if not anomalies:
            return 0.0

        scores = sorted(
            (
                anomaly.score
                * anomaly.confidence
                for anomaly in anomalies
            ),
            reverse=True,
        )

        strongest = scores[0]

        corroboration = min(
            0.15,
            max(
                0,
                len(scores) - 1
            ) * 0.04,
        )

        return _clamp(
            strongest + corroboration
        )

    @staticmethod
    def _health_state(
        severity: SensorSeverity,
        invalid: bool,
    ) -> SensorHealthState:

        if invalid and severity == SensorSeverity.CRITICAL:
            return SensorHealthState.FAILED

        if severity == SensorSeverity.CRITICAL:
            return SensorHealthState.UNRELIABLE

        if severity == SensorSeverity.HIGH:
            return SensorHealthState.UNRELIABLE

        if severity in {
            SensorSeverity.MEDIUM,
            SensorSeverity.LOW,
        }:
            return SensorHealthState.DEGRADED

        return SensorHealthState.HEALTHY

    def _deduplicate(
        self,
        anomalies: Iterable[SensorAnomaly],
    ) -> list[SensorAnomaly]:

        selected: dict[
            SensorAnomalyType,
            SensorAnomaly,
        ] = {}

        for anomaly in anomalies:
            current = selected.get(
                anomaly.anomaly_type
            )

            if current is None:
                selected[
                    anomaly.anomaly_type
                ] = anomaly
                continue

            current_rank = self._SEVERITY_RANK[
                current.severity
            ]

            new_rank = self._SEVERITY_RANK[
                anomaly.severity
            ]

            if (
                new_rank > current_rank
                or (
                    new_rank == current_rank
                    and anomaly.score
                    > current.score
                )
            ):
                selected[
                    anomaly.anomaly_type
                ] = anomaly

        return list(selected.values())


# ============================================================
# Convenience constructors
# ============================================================


def create_sensor_reading(
    *,
    sensor_id: str,
    value: float | None,
    asset_id: str | None = None,
    sensor_type: str | None = None,
    unit: str | None = None,
    site_id: str | None = None,
    zone_id: str | None = None,
    confidence: float = 1.0,
    timestamp: datetime | None = None,
    sequence: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> SensorReading:

    return SensorReading(
        sensor_id=sensor_id,
        timestamp=(
            timestamp
            or datetime.now(timezone.utc)
        ),
        value=value,
        asset_id=asset_id,
        sensor_type=sensor_type,
        unit=unit,
        site_id=site_id,
        zone_id=zone_id,
        confidence=confidence,
        sequence=sequence,
        metadata=metadata or {},
    )


# ============================================================
# Utilities
# ============================================================


def _valid_number(
    value: Any,
) -> bool:

    if value is None:
        return False

    try:
        return isfinite(
            float(value)
        )
    except (TypeError, ValueError):
        return False


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def _median(
    values: Sequence[float],
) -> float:

    ordered = sorted(values)

    length = len(ordered)

    if length == 0:
        raise ValueError(
            "Cannot calculate median "
            "of empty sequence"
        )

    middle = length // 2

    if length % 2:
        return ordered[middle]

    return (
        ordered[middle - 1]
        + ordered[middle]
    ) / 2.0


# ============================================================
# Default detector
# ============================================================


default_sensor_anomaly_detector = (
    SensorAnomalyDetector()
)


__all__ = [
    "SensorAnomalyType",
    "SensorSeverity",
    "SensorHealthState",
    "SensorQuality",
    "SensorReading",
    "SensorLimits",
    "SensorPolicy",
    "SensorEvidence",
    "SensorAnomaly",
    "SensorAssessment",
    "SensorHistory",
    "SensorAnomalyDetector",
    "create_sensor_reading",
    "default_sensor_anomaly_detector",
]