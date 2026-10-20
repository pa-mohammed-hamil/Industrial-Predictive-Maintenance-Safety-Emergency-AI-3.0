# app/ml/anomaly_detection/temperature_anomaly.py

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


class TemperatureAnomalyType(str, Enum):
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"

    RAPID_RISE = "rapid_rise"
    RAPID_DROP = "rapid_drop"

    SPIKE = "spike"
    DROP = "drop"

    SUSTAINED_HIGH = "sustained_high"
    SUSTAINED_LOW = "sustained_low"

    GRADUAL_RISE = "gradual_rise"
    GRADUAL_DROP = "gradual_drop"

    BASELINE_SHIFT = "baseline_shift"
    EXCESSIVE_VARIABILITY = "excessive_variability"

    THERMAL_CYCLING = "thermal_cycling"
    THERMAL_RUNAWAY_PATTERN = "thermal_runaway_pattern"

    SENSOR_SUSPECTED = "sensor_suspected"
    UNKNOWN = "unknown"


class TemperatureSeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TemperatureState(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    ANOMALOUS = "anomalous"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class TemperatureTrend(str, Enum):
    STABLE = "stable"
    RISING = "rising"
    FALLING = "falling"
    RAPIDLY_RISING = "rapidly_rising"
    RAPIDLY_FALLING = "rapidly_falling"
    UNKNOWN = "unknown"


# ============================================================
# Input / Output Models
# ============================================================


@dataclass(slots=True)
class TemperatureReading:
    timestamp: datetime
    temperature_c: float | None

    asset_id: str | None = None
    sensor_id: str | None = None
    component_id: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    ambient_temperature_c: float | None = None

    load_percent: float | None = None
    rpm: float | None = None

    confidence: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(
                tzinfo=timezone.utc
            )

        if (
            self.temperature_c is not None
            and not _valid_number(self.temperature_c)
        ):
            raise ValueError(
                "temperature_c must be finite or None"
            )

        if (
            self.ambient_temperature_c is not None
            and not _valid_number(
                self.ambient_temperature_c
            )
        ):
            raise ValueError(
                "ambient_temperature_c must be finite"
            )


@dataclass(slots=True)
class TemperatureThresholds:
    """
    Generic defaults only.

    Production values should come from the asset/component profile
    or centralized safety rules.
    """

    warning_temperature_c: float = 80.0
    critical_temperature_c: float = 100.0

    low_warning_temperature_c: float | None = None
    low_critical_temperature_c: float | None = None

    rapid_rise_c_per_minute: float = 5.0
    critical_rise_c_per_minute: float = 15.0

    rapid_drop_c_per_minute: float = 8.0

    maximum_temperature_delta_from_ambient_c: float | None = None

    minimum_confidence: float = 0.60


@dataclass(slots=True)
class TemperaturePolicy:
    min_history: int = 15
    max_history: int = 1000

    z_warning: float = 2.0
    z_anomaly: float = 3.0
    z_critical: float = 5.0

    sustained_window: int = 5

    trend_window: int = 10

    drift_window: int = 20
    drift_z_threshold: float = 2.5

    variability_window: int = 20
    variability_std_ratio: float = 0.15

    thermal_cycle_window: int = 30
    thermal_cycle_min_reversals: int = 5

    runaway_window: int = 4
    runaway_min_rate_c_per_minute: float = 5.0
    runaway_acceleration_ratio: float = 1.20

    medium_score: float = 0.40
    high_score: float = 0.70
    critical_score: float = 0.90


@dataclass(slots=True)
class TemperatureEvidence:
    metric: str
    value: float | str | None

    threshold: float | str | None = None

    score: float = 0.0
    description: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TemperatureAnomaly:
    anomaly_id: str

    anomaly_type: TemperatureAnomalyType
    severity: TemperatureSeverity

    score: float
    confidence: float

    timestamp: datetime

    asset_id: str | None
    sensor_id: str | None
    component_id: str | None

    evidence: list[TemperatureEvidence] = field(
        default_factory=list
    )

    requires_investigation: bool = False
    requires_maintenance_review: bool = False
    requires_safety_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TemperatureAssessment:
    assessment_id: str

    timestamp: datetime

    asset_id: str | None
    sensor_id: str | None
    component_id: str | None

    state: TemperatureState
    severity: TemperatureSeverity

    anomaly_score: float
    confidence: float

    temperature_c: float | None

    rate_c_per_minute: float | None
    trend: TemperatureTrend

    baseline_mean_c: float | None
    baseline_std_c: float | None
    z_score: float | None

    anomalies: list[TemperatureAnomaly]
    evidence: list[TemperatureEvidence]

    requires_investigation: bool = False
    requires_maintenance_review: bool = False
    requires_safety_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        return self.severity in {
            TemperatureSeverity.MEDIUM,
            TemperatureSeverity.HIGH,
            TemperatureSeverity.CRITICAL,
        }

    @property
    def is_high_risk(self) -> bool:
        return self.severity in {
            TemperatureSeverity.HIGH,
            TemperatureSeverity.CRITICAL,
        }


# ============================================================
# History
# ============================================================


class TemperatureHistory:
    def __init__(self, max_history: int = 1000) -> None:
        self._readings: deque[TemperatureReading] = deque(
            maxlen=max_history
        )

    def add(
        self,
        reading: TemperatureReading,
    ) -> None:
        self._readings.append(reading)

    @property
    def size(self) -> int:
        return len(self._readings)

    @property
    def readings(self) -> list[TemperatureReading]:
        return list(self._readings)

    @property
    def latest(self) -> TemperatureReading | None:
        if not self._readings:
            return None

        return self._readings[-1]

    def values(self) -> list[float]:
        return [
            float(reading.temperature_c)
            for reading in self._readings
            if _valid_number(reading.temperature_c)
        ]

    def recent(
        self,
        window: int,
    ) -> list[TemperatureReading]:
        return list(self._readings)[-window:]


# ============================================================
# Temperature Anomaly Detector
# ============================================================


class TemperatureAnomalyDetector:
    """
    Statistical + deterministic temperature anomaly detector.

    This module detects thermal abnormalities and predictive
    degradation patterns.

    It does NOT authorize:
        - shutdown
        - emergency isolation
        - fire response
        - evacuation

    Those actions remain under safety_rules.py / decision_engine.
    """

    _SEVERITY_RANK = {
        TemperatureSeverity.NORMAL: 0,
        TemperatureSeverity.LOW: 1,
        TemperatureSeverity.MEDIUM: 2,
        TemperatureSeverity.HIGH: 3,
        TemperatureSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        thresholds: TemperatureThresholds | None = None,
        policy: TemperaturePolicy | None = None,
    ) -> None:

        self.thresholds = (
            thresholds
            or TemperatureThresholds()
        )

        self.policy = (
            policy
            or TemperaturePolicy()
        )

        self._histories: dict[
            str,
            TemperatureHistory,
        ] = {}

    # ========================================================
    # Public API
    # ========================================================

    def assess(
        self,
        reading: TemperatureReading,
        *,
        update_history: bool = True,
    ) -> TemperatureAssessment:

        history = self._history_for(reading)

        if (
            reading.temperature_c is None
            or reading.confidence
            < self.thresholds.minimum_confidence
        ):
            return self._insufficient_assessment(
                reading,
                history,
            )

        anomalies: list[TemperatureAnomaly] = []

        anomalies.extend(
            self._check_absolute_temperature(
                reading
            )
        )

        rate = self._temperature_rate(
            reading,
            history.latest,
        )

        anomalies.extend(
            self._check_rate(
                reading,
                rate,
            )
        )

        anomalies.extend(
            self._check_ambient_delta(
                reading
            )
        )

        if history.size >= self.policy.min_history:

            anomalies.extend(
                self._check_statistical_deviation(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_sustained_temperature(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_baseline_shift(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_variability(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_gradual_trend(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_thermal_cycling(
                    reading,
                    history,
                )
            )

            anomalies.extend(
                self._check_runaway_pattern(
                    reading,
                    history,
                )
            )

        anomalies = self._deduplicate(
            anomalies
        )

        severity = self._overall_severity(
            anomalies
        )

        score = self._overall_score(
            anomalies
        )

        trend = self._trend(
            reading,
            history,
            rate,
        )

        baseline_mean, baseline_std, z_score = (
            self._baseline_statistics(
                reading,
                history,
            )
        )

        evidence = [
            evidence
            for anomaly in anomalies
            for evidence in anomaly.evidence
        ]

        safety_review = any(
            anomaly.requires_safety_review
            for anomaly in anomalies
        )

        maintenance_review = any(
            anomaly.requires_maintenance_review
            for anomaly in anomalies
        )

        assessment = TemperatureAssessment(
            assessment_id=str(uuid4()),
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            component_id=reading.component_id,
            state=self._state(severity),
            severity=severity,
            anomaly_score=score,
            confidence=reading.confidence,
            temperature_c=reading.temperature_c,
            rate_c_per_minute=rate,
            trend=trend,
            baseline_mean_c=baseline_mean,
            baseline_std_c=baseline_std,
            z_score=z_score,
            anomalies=anomalies,
            evidence=evidence,
            requires_investigation=bool(anomalies),
            requires_maintenance_review=maintenance_review,
            requires_safety_review=safety_review,
            metadata={
                "history_size": history.size,
                "site_id": reading.site_id,
                "zone_id": reading.zone_id,
                "load_percent": reading.load_percent,
                "rpm": reading.rpm,
            },
        )

        # Avoid contaminating baseline with clearly abnormal data.
        if (
            update_history
            and severity
            not in {
                TemperatureSeverity.HIGH,
                TemperatureSeverity.CRITICAL,
            }
        ):
            history.add(reading)

        return assessment

    def detect(
        self,
        reading: TemperatureReading,
    ) -> list[TemperatureAnomaly]:

        return self.assess(
            reading
        ).anomalies

    def assess_many(
        self,
        readings: Iterable[TemperatureReading],
    ) -> list[TemperatureAssessment]:

        ordered = sorted(
            readings,
            key=lambda item: item.timestamp,
        )

        return [
            self.assess(reading)
            for reading in ordered
        ]

    # ========================================================
    # Absolute temperature
    # ========================================================

    def _check_absolute_temperature(
        self,
        reading: TemperatureReading,
    ) -> list[TemperatureAnomaly]:

        value = float(reading.temperature_c)

        t = self.thresholds

        if value >= t.critical_temperature_c:
            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.HIGH_TEMPERATURE,
                    TemperatureSeverity.CRITICAL,
                    1.0,
                    [
                        TemperatureEvidence(
                            metric="temperature_c",
                            value=value,
                            threshold=t.critical_temperature_c,
                            score=1.0,
                            description=(
                                "Temperature reached "
                                "critical configured level"
                            ),
                        )
                    ],
                    safety=True,
                    maintenance=True,
                )
            ]

        if value >= t.warning_temperature_c:

            score = _normalize(
                value,
                t.warning_temperature_c,
                t.critical_temperature_c,
            )

            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.HIGH_TEMPERATURE,
                    TemperatureSeverity.HIGH,
                    max(0.70, score),
                    [
                        TemperatureEvidence(
                            metric="temperature_c",
                            value=value,
                            threshold=t.warning_temperature_c,
                            score=score,
                            description=(
                                "Temperature exceeds "
                                "warning level"
                            ),
                        )
                    ],
                    safety=True,
                    maintenance=True,
                )
            ]

        if (
            t.low_critical_temperature_c is not None
            and value <= t.low_critical_temperature_c
        ):
            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.LOW_TEMPERATURE,
                    TemperatureSeverity.CRITICAL,
                    1.0,
                    [
                        TemperatureEvidence(
                            metric="temperature_c",
                            value=value,
                            threshold=t.low_critical_temperature_c,
                            score=1.0,
                            description=(
                                "Temperature below critical "
                                "lower operating limit"
                            ),
                        )
                    ],
                    safety=True,
                )
            ]

        if (
            t.low_warning_temperature_c is not None
            and value <= t.low_warning_temperature_c
        ):
            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.LOW_TEMPERATURE,
                    TemperatureSeverity.HIGH,
                    0.75,
                    [
                        TemperatureEvidence(
                            metric="temperature_c",
                            value=value,
                            threshold=t.low_warning_temperature_c,
                            score=0.75,
                            description=(
                                "Temperature below configured "
                                "lower warning limit"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            ]

        return []

    # ========================================================
    # Rate of change
    # ========================================================

    def _temperature_rate(
        self,
        current: TemperatureReading,
        previous: TemperatureReading | None,
    ) -> float | None:

        if (
            previous is None
            or previous.temperature_c is None
        ):
            return None

        seconds = (
            current.timestamp
            - previous.timestamp
        ).total_seconds()

        if seconds <= 0:
            return None

        delta = (
            float(current.temperature_c)
            - float(previous.temperature_c)
        )

        return delta / seconds * 60.0

    def _check_rate(
        self,
        reading: TemperatureReading,
        rate: float | None,
    ) -> list[TemperatureAnomaly]:

        if rate is None:
            return []

        thresholds = self.thresholds

        if (
            rate
            >= thresholds.critical_rise_c_per_minute
        ):
            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.RAPID_RISE,
                    TemperatureSeverity.CRITICAL,
                    1.0,
                    [
                        TemperatureEvidence(
                            metric="temperature_rate_c_per_min",
                            value=rate,
                            threshold=(
                                thresholds
                                .critical_rise_c_per_minute
                            ),
                            score=1.0,
                            description=(
                                "Critical temperature rise rate"
                            ),
                        )
                    ],
                    safety=True,
                    maintenance=True,
                )
            ]

        if (
            rate
            >= thresholds.rapid_rise_c_per_minute
        ):
            score = _normalize(
                rate,
                thresholds.rapid_rise_c_per_minute,
                thresholds.critical_rise_c_per_minute,
            )

            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.RAPID_RISE,
                    TemperatureSeverity.HIGH,
                    max(0.70, score),
                    [
                        TemperatureEvidence(
                            metric="temperature_rate_c_per_min",
                            value=rate,
                            threshold=(
                                thresholds
                                .rapid_rise_c_per_minute
                            ),
                            score=score,
                            description=(
                                "Rapid temperature rise detected"
                            ),
                        )
                    ],
                    safety=True,
                    maintenance=True,
                )
            ]

        if (
            rate
            <= -thresholds.rapid_drop_c_per_minute
        ):
            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.RAPID_DROP,
                    TemperatureSeverity.MEDIUM,
                    0.60,
                    [
                        TemperatureEvidence(
                            metric="temperature_rate_c_per_min",
                            value=rate,
                            threshold=(
                                -thresholds
                                .rapid_drop_c_per_minute
                            ),
                            score=0.60,
                            description=(
                                "Rapid temperature drop detected"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            ]

        return []

    # ========================================================
    # Ambient-relative temperature
    # ========================================================

    def _check_ambient_delta(
        self,
        reading: TemperatureReading,
    ) -> list[TemperatureAnomaly]:

        limit = (
            self.thresholds
            .maximum_temperature_delta_from_ambient_c
        )

        if (
            limit is None
            or reading.ambient_temperature_c is None
        ):
            return []

        delta = (
            float(reading.temperature_c)
            - float(reading.ambient_temperature_c)
        )

        if delta <= limit:
            return []

        score = _clamp(
            delta / max(limit * 2.0, 1e-9)
        )

        return [
            self._anomaly(
                reading,
                TemperatureAnomalyType.HIGH_TEMPERATURE,
                TemperatureSeverity.MEDIUM,
                max(0.50, score),
                [
                    TemperatureEvidence(
                        metric="temperature_delta_ambient_c",
                        value=delta,
                        threshold=limit,
                        score=score,
                        description=(
                            "Temperature is unusually high "
                            "relative to ambient conditions"
                        ),
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Statistical anomaly
    # ========================================================

    def _check_statistical_deviation(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> list[TemperatureAnomaly]:

        values = history.values()

        if len(values) < self.policy.min_history:
            return []

        mu = mean(values)
        sigma = pstdev(values)

        if sigma <= 1e-12:
            return []

        value = float(reading.temperature_c)

        z = (value - mu) / sigma
        absolute_z = abs(z)

        if absolute_z < self.policy.z_anomaly:
            return []

        if absolute_z >= self.policy.z_critical:
            severity = TemperatureSeverity.HIGH
            score = 0.90
        else:
            severity = TemperatureSeverity.MEDIUM
            score = _normalize(
                absolute_z,
                self.policy.z_anomaly,
                self.policy.z_critical,
            )

        anomaly_type = (
            TemperatureAnomalyType.SPIKE
            if z > 0
            else TemperatureAnomalyType.DROP
        )

        return [
            self._anomaly(
                reading,
                anomaly_type,
                severity,
                max(0.55, score),
                [
                    TemperatureEvidence(
                        metric="temperature_z_score",
                        value=z,
                        threshold=self.policy.z_anomaly,
                        score=score,
                        description=(
                            "Temperature differs significantly "
                            "from historical baseline"
                        ),
                        metadata={
                            "baseline_mean": mu,
                            "baseline_std": sigma,
                        },
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Sustained abnormal temperature
    # ========================================================

    def _check_sustained_temperature(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> list[TemperatureAnomaly]:

        recent = history.recent(
            self.policy.sustained_window - 1
        )

        values = [
            float(item.temperature_c)
            for item in recent
            if item.temperature_c is not None
        ]

        values.append(
            float(reading.temperature_c)
        )

        if (
            len(values)
            < self.policy.sustained_window
        ):
            return []

        warning = (
            self.thresholds.warning_temperature_c
        )

        if all(
            value >= warning
            for value in values
        ):
            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.SUSTAINED_HIGH,
                    TemperatureSeverity.HIGH,
                    0.85,
                    [
                        TemperatureEvidence(
                            metric="sustained_temperature",
                            value=mean(values),
                            threshold=warning,
                            score=0.85,
                            description=(
                                "Temperature remained above "
                                "warning level across multiple "
                                "readings"
                            ),
                        )
                    ],
                    safety=True,
                    maintenance=True,
                )
            ]

        return []

    # ========================================================
    # Baseline shift / drift
    # ========================================================

    def _check_baseline_shift(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> list[TemperatureAnomaly]:

        values = history.values()
        window = self.policy.drift_window

        if len(values) < window * 2:
            return []

        reference = values[:-window]
        recent = values[-window + 1 :] + [
            float(reading.temperature_c)
        ]

        if not reference:
            return []

        reference_mean = mean(reference)
        reference_std = pstdev(reference)

        if reference_std <= 1e-12:
            return []

        recent_mean = mean(recent)

        shift_z = (
            recent_mean - reference_mean
        ) / reference_std

        if (
            abs(shift_z)
            < self.policy.drift_z_threshold
        ):
            return []

        anomaly_type = (
            TemperatureAnomalyType.GRADUAL_RISE
            if shift_z > 0
            else TemperatureAnomalyType.GRADUAL_DROP
        )

        score = _clamp(
            abs(shift_z)
            / (
                self.policy.drift_z_threshold
                * 2.0
            )
        )

        return [
            self._anomaly(
                reading,
                anomaly_type,
                TemperatureSeverity.MEDIUM,
                max(0.55, score),
                [
                    TemperatureEvidence(
                        metric="baseline_shift_z",
                        value=shift_z,
                        threshold=(
                            self.policy
                            .drift_z_threshold
                        ),
                        score=score,
                        description=(
                            "Long-term temperature baseline "
                            "has shifted"
                        ),
                        metadata={
                            "reference_mean":
                                reference_mean,
                            "recent_mean":
                                recent_mean,
                        },
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Excessive variability
    # ========================================================

    def _check_variability(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> list[TemperatureAnomaly]:

        recent = history.recent(
            self.policy.variability_window - 1
        )

        values = [
            float(item.temperature_c)
            for item in recent
            if item.temperature_c is not None
        ]

        values.append(
            float(reading.temperature_c)
        )

        if (
            len(values)
            < self.policy.variability_window
        ):
            return []

        mu = mean(values)
        sigma = pstdev(values)

        denominator = max(
            abs(mu),
            1.0,
        )

        variability_ratio = (
            sigma / denominator
        )

        if (
            variability_ratio
            <= self.policy.variability_std_ratio
        ):
            return []

        score = _clamp(
            variability_ratio
            / (
                self.policy
                .variability_std_ratio
                * 2.0
            )
        )

        return [
            self._anomaly(
                reading,
                TemperatureAnomalyType.EXCESSIVE_VARIABILITY,
                TemperatureSeverity.MEDIUM,
                max(0.50, score),
                [
                    TemperatureEvidence(
                        metric="temperature_variability_ratio",
                        value=variability_ratio,
                        threshold=(
                            self.policy
                            .variability_std_ratio
                        ),
                        score=score,
                        description=(
                            "Temperature signal has excessive "
                            "short-term variability"
                        ),
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Gradual trend
    # ========================================================

    def _check_gradual_trend(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> list[TemperatureAnomaly]:

        recent = history.recent(
            self.policy.trend_window - 1
        ) + [reading]

        points = [
            item
            for item in recent
            if item.temperature_c is not None
        ]

        if len(points) < self.policy.trend_window:
            return []

        slope = _linear_temperature_slope(
            points
        )

        # Celsius per minute.
        if (
            slope
            >= self.thresholds
            .rapid_rise_c_per_minute * 0.25
        ):
            score = _clamp(
                slope
                / max(
                    self.thresholds
                    .rapid_rise_c_per_minute,
                    1e-9,
                )
            )

            return [
                self._anomaly(
                    reading,
                    TemperatureAnomalyType.GRADUAL_RISE,
                    TemperatureSeverity.MEDIUM,
                    max(0.45, score),
                    [
                        TemperatureEvidence(
                            metric="temperature_trend_c_per_min",
                            value=slope,
                            threshold=(
                                self.thresholds
                                .rapid_rise_c_per_minute
                                * 0.25
                            ),
                            score=score,
                            description=(
                                "Persistent upward thermal "
                                "trend detected"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            ]

        return []

    # ========================================================
    # Thermal cycling
    # ========================================================

    def _check_thermal_cycling(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> list[TemperatureAnomaly]:

        recent = history.recent(
            self.policy.thermal_cycle_window - 1
        ) + [reading]

        values = [
            float(item.temperature_c)
            for item in recent
            if item.temperature_c is not None
        ]

        if (
            len(values)
            < self.policy.thermal_cycle_window
        ):
            return []

        changes = [
            values[index] - values[index - 1]
            for index in range(1, len(values))
        ]

        signs = [
            1 if change > 0 else -1
            for change in changes
            if abs(change) > 1e-6
        ]

        reversals = sum(
            1
            for index in range(1, len(signs))
            if signs[index] != signs[index - 1]
        )

        if (
            reversals
            < self.policy.thermal_cycle_min_reversals
        ):
            return []

        amplitude = max(values) - min(values)

        if amplitude < 5.0:
            return []

        score = _clamp(
            reversals
            / max(
                self.policy
                .thermal_cycle_min_reversals
                * 2,
                1,
            )
        )

        return [
            self._anomaly(
                reading,
                TemperatureAnomalyType.THERMAL_CYCLING,
                TemperatureSeverity.MEDIUM,
                max(0.50, score),
                [
                    TemperatureEvidence(
                        metric="thermal_cycle_reversals",
                        value=float(reversals),
                        threshold=float(
                            self.policy
                            .thermal_cycle_min_reversals
                        ),
                        score=score,
                        description=(
                            "Repeated heating/cooling cycles "
                            "detected"
                        ),
                        metadata={
                            "temperature_amplitude_c":
                                amplitude,
                        },
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Thermal runaway pattern
    # ========================================================

    def _check_runaway_pattern(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> list[TemperatureAnomaly]:

        recent = history.recent(
            self.policy.runaway_window - 1
        ) + [reading]

        if (
            len(recent)
            < self.policy.runaway_window
        ):
            return []

        rates: list[float] = []

        for previous, current in zip(
            recent,
            recent[1:],
        ):
            rate = self._temperature_rate(
                current,
                previous,
            )

            if rate is None:
                continue

            rates.append(rate)

        if len(rates) < 3:
            return []

        if any(
            rate
            < self.policy
            .runaway_min_rate_c_per_minute
            for rate in rates
        ):
            return []

        accelerating = all(
            rates[index]
            >= rates[index - 1]
            * self.policy
            .runaway_acceleration_ratio
            for index in range(
                1,
                len(rates),
            )
        )

        if not accelerating:
            return []

        return [
            self._anomaly(
                reading,
                TemperatureAnomalyType.THERMAL_RUNAWAY_PATTERN,
                TemperatureSeverity.CRITICAL,
                1.0,
                [
                    TemperatureEvidence(
                        metric="temperature_acceleration",
                        value=rates[-1],
                        threshold=(
                            self.policy
                            .runaway_min_rate_c_per_minute
                        ),
                        score=1.0,
                        description=(
                            "Accelerating thermal-rise "
                            "pattern detected"
                        ),
                        metadata={
                            "rates_c_per_minute":
                                rates,
                        },
                    )
                ],
                safety=True,
                maintenance=True,
            )
        ]

    # ========================================================
    # Baseline / trend
    # ========================================================

    def _baseline_statistics(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
    ]:

        values = history.values()

        if not values:
            return None, None, None

        mu = mean(values)

        sigma = (
            pstdev(values)
            if len(values) > 1
            else 0.0
        )

        if sigma <= 1e-12:
            return mu, sigma, None

        z = (
            float(reading.temperature_c)
            - mu
        ) / sigma

        return mu, sigma, z

    def _trend(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
        rate: float | None,
    ) -> TemperatureTrend:

        if rate is None:
            return TemperatureTrend.UNKNOWN

        if (
            rate
            >= self.thresholds
            .rapid_rise_c_per_minute
        ):
            return TemperatureTrend.RAPIDLY_RISING

        if (
            rate
            <= -self.thresholds
            .rapid_drop_c_per_minute
        ):
            return TemperatureTrend.RAPIDLY_FALLING

        if rate > 0.1:
            return TemperatureTrend.RISING

        if rate < -0.1:
            return TemperatureTrend.FALLING

        return TemperatureTrend.STABLE

    # ========================================================
    # Helpers
    # ========================================================

    def _history_for(
        self,
        reading: TemperatureReading,
    ) -> TemperatureHistory:

        key = (
            reading.sensor_id
            or reading.component_id
            or reading.asset_id
            or "default"
        )

        history = self._histories.get(key)

        if history is None:
            history = TemperatureHistory(
                self.policy.max_history
            )

            self._histories[key] = history

        return history

    def _anomaly(
        self,
        reading: TemperatureReading,
        anomaly_type: TemperatureAnomalyType,
        severity: TemperatureSeverity,
        score: float,
        evidence: list[TemperatureEvidence],
        *,
        maintenance: bool = False,
        safety: bool = False,
    ) -> TemperatureAnomaly:

        return TemperatureAnomaly(
            anomaly_id=str(uuid4()),
            anomaly_type=anomaly_type,
            severity=severity,
            score=_clamp(score),
            confidence=reading.confidence,
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            component_id=reading.component_id,
            evidence=evidence,
            requires_investigation=True,
            requires_maintenance_review=maintenance,
            requires_safety_review=safety,
            metadata={
                "site_id": reading.site_id,
                "zone_id": reading.zone_id,
            },
        )

    def _overall_severity(
        self,
        anomalies: Sequence[TemperatureAnomaly],
    ) -> TemperatureSeverity:

        if not anomalies:
            return TemperatureSeverity.NORMAL

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
        anomalies: Sequence[TemperatureAnomaly],
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

        corroboration_bonus = min(
            0.15,
            max(
                0,
                len(scores) - 1
            ) * 0.04,
        )

        return _clamp(
            strongest
            + corroboration_bonus
        )

    @staticmethod
    def _state(
        severity: TemperatureSeverity,
    ) -> TemperatureState:

        mapping = {
            TemperatureSeverity.NORMAL:
                TemperatureState.NORMAL,

            TemperatureSeverity.LOW:
                TemperatureState.ELEVATED,

            TemperatureSeverity.MEDIUM:
                TemperatureState.ANOMALOUS,

            TemperatureSeverity.HIGH:
                TemperatureState.ANOMALOUS,

            TemperatureSeverity.CRITICAL:
                TemperatureState.CRITICAL,
        }

        return mapping[severity]

    def _deduplicate(
        self,
        anomalies: Sequence[TemperatureAnomaly],
    ) -> list[TemperatureAnomaly]:

        selected: dict[
            TemperatureAnomalyType,
            TemperatureAnomaly,
        ] = {}

        for anomaly in anomalies:

            existing = selected.get(
                anomaly.anomaly_type
            )

            if existing is None:
                selected[
                    anomaly.anomaly_type
                ] = anomaly
                continue

            existing_rank = self._SEVERITY_RANK[
                existing.severity
            ]

            new_rank = self._SEVERITY_RANK[
                anomaly.severity
            ]

            if (
                new_rank > existing_rank
                or (
                    new_rank == existing_rank
                    and anomaly.score
                    > existing.score
                )
            ):
                selected[
                    anomaly.anomaly_type
                ] = anomaly

        return list(
            selected.values()
        )

    def _insufficient_assessment(
        self,
        reading: TemperatureReading,
        history: TemperatureHistory,
    ) -> TemperatureAssessment:

        return TemperatureAssessment(
            assessment_id=str(uuid4()),
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            component_id=reading.component_id,
            state=TemperatureState.INSUFFICIENT_DATA,
            severity=TemperatureSeverity.NORMAL,
            anomaly_score=0.0,
            confidence=reading.confidence,
            temperature_c=reading.temperature_c,
            rate_c_per_minute=None,
            trend=TemperatureTrend.UNKNOWN,
            baseline_mean_c=None,
            baseline_std_c=None,
            z_score=None,
            anomalies=[],
            evidence=[
                TemperatureEvidence(
                    metric="temperature",
                    value=reading.temperature_c,
                    score=0.0,
                    description=(
                        "Temperature reading missing or "
                        "confidence below minimum"
                    ),
                )
            ],
            metadata={
                "history_size":
                    history.size,
            },
        )


# ============================================================
# Linear trend utility
# ============================================================


def _linear_temperature_slope(
    readings: Sequence[TemperatureReading],
) -> float:
    """
    Least-squares temperature slope in Celsius/minute.
    """

    if len(readings) < 2:
        return 0.0

    start = readings[0].timestamp

    x: list[float] = []
    y: list[float] = []

    for reading in readings:

        if reading.temperature_c is None:
            continue

        minutes = (
            reading.timestamp - start
        ).total_seconds() / 60.0

        x.append(minutes)
        y.append(
            float(reading.temperature_c)
        )

    if len(x) < 2:
        return 0.0

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum(
        (xi - mean_x)
        * (yi - mean_y)
        for xi, yi in zip(x, y)
    )

    denominator = sum(
        (xi - mean_x) ** 2
        for xi in x
    )

    if denominator <= 1e-12:
        return 0.0

    return numerator / denominator


# ============================================================
# Convenience API
# ============================================================


def create_temperature_reading(
    temperature_c: float,
    *,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    ambient_temperature_c: float | None = None,
    load_percent: float | None = None,
    rpm: float | None = None,
    confidence: float = 1.0,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> TemperatureReading:

    return TemperatureReading(
        timestamp=(
            timestamp
            or datetime.now(timezone.utc)
        ),
        temperature_c=temperature_c,
        asset_id=asset_id,
        sensor_id=sensor_id,
        component_id=component_id,
        ambient_temperature_c=(
            ambient_temperature_c
        ),
        load_percent=load_percent,
        rpm=rpm,
        confidence=confidence,
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


def _normalize(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    if maximum <= minimum:
        return 1.0

    return _clamp(
        (value - minimum)
        / (maximum - minimum)
    )


# ============================================================
# Default detector
# ============================================================


default_temperature_anomaly_detector = (
    TemperatureAnomalyDetector()
)


__all__ = [
    "TemperatureAnomalyType",
    "TemperatureSeverity",
    "TemperatureState",
    "TemperatureTrend",
    "TemperatureReading",
    "TemperatureThresholds",
    "TemperaturePolicy",
    "TemperatureEvidence",
    "TemperatureAnomaly",
    "TemperatureAssessment",
    "TemperatureHistory",
    "TemperatureAnomalyDetector",
    "create_temperature_reading",
    "default_temperature_anomaly_detector",
]