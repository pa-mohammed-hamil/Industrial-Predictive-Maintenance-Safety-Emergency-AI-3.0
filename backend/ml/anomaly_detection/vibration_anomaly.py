# app/ml/anomaly_detection/vibration_anomaly.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


# ============================================================
# Enums
# ============================================================


class VibrationAnomalyType(str, Enum):
    HIGH_RMS = "high_rms"
    HIGH_PEAK = "high_peak"
    HIGH_CREST_FACTOR = "high_crest_factor"
    HIGH_KURTOSIS = "high_kurtosis"

    SPIKE = "spike"
    DROP = "drop"

    RAPID_INCREASE = "rapid_increase"
    GRADUAL_INCREASE = "gradual_increase"
    BASELINE_SHIFT = "baseline_shift"

    EXCESSIVE_VARIABILITY = "excessive_variability"

    IMBALANCE_SUSPECTED = "imbalance_suspected"
    MISALIGNMENT_SUSPECTED = "misalignment_suspected"
    LOOSENESS_SUSPECTED = "looseness_suspected"

    BEARING_FAULT_SUSPECTED = "bearing_fault_suspected"
    GEAR_FAULT_SUSPECTED = "gear_fault_suspected"

    RESONANCE_SUSPECTED = "resonance_suspected"
    SENSOR_SUSPECTED = "sensor_suspected"

    MULTI_INDICATOR = "multi_indicator"
    UNKNOWN = "unknown"


class VibrationSeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VibrationState(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    ANOMALOUS = "anomalous"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class VibrationTrend(str, Enum):
    STABLE = "stable"
    RISING = "rising"
    FALLING = "falling"
    RAPIDLY_RISING = "rapidly_rising"
    UNKNOWN = "unknown"


# ============================================================
# Input Models
# ============================================================


@dataclass(slots=True)
class FrequencyBand:
    """
    Frequency-band energy produced by FFT/spectral processing.

    Example:
        FrequencyBand(
            name="1x_rpm",
            low_hz=28,
            high_hz=32,
            amplitude=4.2,
        )
    """

    name: str
    low_hz: float
    high_hz: float
    amplitude: float

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.low_hz < 0:
            raise ValueError("low_hz cannot be negative")

        if self.high_hz < self.low_hz:
            raise ValueError(
                "high_hz must be >= low_hz"
            )

        if not _valid_number(self.amplitude):
            raise ValueError(
                "amplitude must be finite"
            )


@dataclass(slots=True)
class VibrationReading:
    timestamp: datetime

    rms_mm_s: float | None = None
    peak_mm_s: float | None = None

    acceleration_rms_g: float | None = None
    acceleration_peak_g: float | None = None

    crest_factor: float | None = None
    kurtosis: float | None = None

    rpm: float | None = None
    load_percent: float | None = None

    frequency_bands: Mapping[str, float] = field(
        default_factory=dict
    )

    asset_id: str | None = None
    sensor_id: str | None = None
    component_id: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    axis: str | None = None

    confidence: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(
                tzinfo=timezone.utc
            )

        fields = (
            "rms_mm_s",
            "peak_mm_s",
            "acceleration_rms_g",
            "acceleration_peak_g",
            "crest_factor",
            "kurtosis",
            "rpm",
            "load_percent",
        )

        for name in fields:
            value = getattr(self, name)

            if (
                value is not None
                and not _valid_number(value)
            ):
                raise ValueError(
                    f"{name} must be finite or None"
                )


# ============================================================
# Configuration
# ============================================================


@dataclass(slots=True)
class VibrationThresholds:
    """
    Generic demo defaults.

    Production thresholds must be configured per machine,
    component, mounting orientation, sensor type and operating
    regime rather than treating these as universal standards.
    """

    rms_warning_mm_s: float = 4.5
    rms_critical_mm_s: float = 11.0

    peak_warning_mm_s: float = 8.0
    peak_critical_mm_s: float = 20.0

    crest_factor_warning: float = 4.0
    crest_factor_critical: float = 6.0

    kurtosis_warning: float = 4.0
    kurtosis_critical: float = 8.0

    rapid_rms_increase_per_minute: float = 1.5
    critical_rms_increase_per_minute: float = 4.0

    minimum_confidence: float = 0.60


@dataclass(slots=True)
class VibrationPolicy:
    min_history: int = 15
    max_history: int = 1000

    z_anomaly: float = 3.0
    z_critical: float = 5.0

    trend_window: int = 10
    drift_window: int = 20

    drift_z_threshold: float = 2.5

    variability_window: int = 20
    variability_ratio: float = 0.35

    # Relative spectral energy thresholds.
    one_x_warning_ratio: float = 0.40
    two_x_warning_ratio: float = 0.30

    bearing_band_warning_ratio: float = 0.25
    bearing_band_critical_ratio: float = 0.45

    gear_band_warning_ratio: float = 0.30
    gear_band_critical_ratio: float = 0.50

    multi_indicator_count: int = 3

    medium_score: float = 0.40
    high_score: float = 0.70
    critical_score: float = 0.90


# ============================================================
# Evidence / Results
# ============================================================


@dataclass(slots=True)
class VibrationEvidence:
    metric: str
    value: float | str | None

    threshold: float | str | None = None

    score: float = 0.0
    description: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VibrationAnomaly:
    anomaly_id: str

    anomaly_type: VibrationAnomalyType
    severity: VibrationSeverity

    score: float
    confidence: float

    timestamp: datetime

    asset_id: str | None
    sensor_id: str | None
    component_id: str | None

    evidence: list[VibrationEvidence] = field(
        default_factory=list
    )

    requires_investigation: bool = False
    requires_maintenance_review: bool = False
    requires_safety_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VibrationAssessment:
    assessment_id: str

    timestamp: datetime

    asset_id: str | None
    sensor_id: str | None
    component_id: str | None

    state: VibrationState
    severity: VibrationSeverity

    anomaly_score: float
    confidence: float

    rms_mm_s: float | None

    rms_rate_per_minute: float | None
    trend: VibrationTrend

    baseline_mean_rms: float | None
    baseline_std_rms: float | None
    rms_z_score: float | None

    anomalies: list[VibrationAnomaly]
    evidence: list[VibrationEvidence]

    suspected_faults: list[str]

    requires_investigation: bool = False
    requires_maintenance_review: bool = False
    requires_safety_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        return self.severity in {
            VibrationSeverity.MEDIUM,
            VibrationSeverity.HIGH,
            VibrationSeverity.CRITICAL,
        }

    @property
    def is_high_risk(self) -> bool:
        return self.severity in {
            VibrationSeverity.HIGH,
            VibrationSeverity.CRITICAL,
        }


# ============================================================
# History
# ============================================================


class VibrationHistory:

    def __init__(
        self,
        max_history: int = 1000,
    ) -> None:

        self._readings: deque[
            VibrationReading
        ] = deque(
            maxlen=max_history
        )

    def add(
        self,
        reading: VibrationReading,
    ) -> None:
        self._readings.append(reading)

    @property
    def size(self) -> int:
        return len(self._readings)

    @property
    def readings(
        self,
    ) -> list[VibrationReading]:
        return list(self._readings)

    @property
    def latest(
        self,
    ) -> VibrationReading | None:

        if not self._readings:
            return None

        return self._readings[-1]

    def rms_values(self) -> list[float]:

        return [
            float(reading.rms_mm_s)
            for reading in self._readings
            if _valid_number(
                reading.rms_mm_s
            )
        ]

    def recent(
        self,
        window: int,
    ) -> list[VibrationReading]:

        return list(
            self._readings
        )[-window:]


# ============================================================
# Detector
# ============================================================


class VibrationAnomalyDetector:
    """
    Industrial vibration anomaly detector.

    Detects:
      - RMS/peak threshold anomalies
      - statistical vibration deviations
      - rapid increases
      - long-term degradation
      - excessive variability
      - crest-factor / kurtosis abnormalities
      - imbalance indicators
      - misalignment indicators
      - looseness indicators
      - bearing spectral abnormalities
      - gear spectral abnormalities

    This detector provides condition-monitoring evidence.

    It does NOT directly authorize machine shutdown.
    """

    _SEVERITY_RANK = {
        VibrationSeverity.NORMAL: 0,
        VibrationSeverity.LOW: 1,
        VibrationSeverity.MEDIUM: 2,
        VibrationSeverity.HIGH: 3,
        VibrationSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        thresholds: VibrationThresholds | None = None,
        policy: VibrationPolicy | None = None,
    ) -> None:

        self.thresholds = (
            thresholds
            or VibrationThresholds()
        )

        self.policy = (
            policy
            or VibrationPolicy()
        )

        self._histories: dict[
            str,
            VibrationHistory,
        ] = {}

    # ========================================================
    # Public API
    # ========================================================

    def assess(
        self,
        reading: VibrationReading,
        *,
        update_history: bool = True,
    ) -> VibrationAssessment:

        history = self._history_for(
            reading
        )

        if (
            reading.confidence
            < self.thresholds.minimum_confidence
        ):
            return self._insufficient(
                reading,
                history,
                "low_confidence",
            )

        if not self._has_measurement(reading):
            return self._insufficient(
                reading,
                history,
                "missing_vibration_measurement",
            )

        anomalies: list[
            VibrationAnomaly
        ] = []

        anomalies.extend(
            self._check_rms(reading)
        )

        anomalies.extend(
            self._check_peak(reading)
        )

        anomalies.extend(
            self._check_crest_factor(reading)
        )

        anomalies.extend(
            self._check_kurtosis(reading)
        )

        rate = self._rms_rate(
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
            self._check_spectral_faults(
                reading
            )
        )

        if (
            history.size
            >= self.policy.min_history
        ):
            anomalies.extend(
                self._check_statistical_rms(
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
                self._check_trend(
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

        anomalies = self._deduplicate(
            anomalies
        )

        anomalies = self._apply_multi_indicator(
            reading,
            anomalies,
        )

        severity = self._overall_severity(
            anomalies
        )

        score = self._overall_score(
            anomalies
        )

        trend = self._trend_state(
            rate
        )

        baseline_mean, baseline_std, z = (
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

        suspected_faults = (
            self._suspected_faults(
                anomalies
            )
        )

        maintenance_review = any(
            anomaly.requires_maintenance_review
            for anomaly in anomalies
        )

        safety_review = any(
            anomaly.requires_safety_review
            for anomaly in anomalies
        )

        assessment = VibrationAssessment(
            assessment_id=str(uuid4()),
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            component_id=reading.component_id,
            state=self._state(severity),
            severity=severity,
            anomaly_score=score,
            confidence=reading.confidence,
            rms_mm_s=reading.rms_mm_s,
            rms_rate_per_minute=rate,
            trend=trend,
            baseline_mean_rms=baseline_mean,
            baseline_std_rms=baseline_std,
            rms_z_score=z,
            anomalies=anomalies,
            evidence=evidence,
            suspected_faults=suspected_faults,
            requires_investigation=bool(
                anomalies
            ),
            requires_maintenance_review=(
                maintenance_review
            ),
            requires_safety_review=(
                safety_review
            ),
            metadata={
                "history_size":
                    history.size,
                "rpm":
                    reading.rpm,
                "load_percent":
                    reading.load_percent,
                "axis":
                    reading.axis,
                "site_id":
                    reading.site_id,
                "zone_id":
                    reading.zone_id,
            },
        )

        # Avoid contaminating the normal baseline with
        # clearly abnormal vibration data.
        if (
            update_history
            and severity
            not in {
                VibrationSeverity.HIGH,
                VibrationSeverity.CRITICAL,
            }
        ):
            history.add(reading)

        return assessment

    def detect(
        self,
        reading: VibrationReading,
    ) -> list[VibrationAnomaly]:

        return self.assess(
            reading
        ).anomalies

    def assess_many(
        self,
        readings: Iterable[
            VibrationReading
        ],
    ) -> list[VibrationAssessment]:

        ordered = sorted(
            readings,
            key=lambda item:
                item.timestamp,
        )

        return [
            self.assess(reading)
            for reading in ordered
        ]

    # ========================================================
    # RMS
    # ========================================================

    def _check_rms(
        self,
        reading: VibrationReading,
    ) -> list[VibrationAnomaly]:

        if not _valid_number(
            reading.rms_mm_s
        ):
            return []

        value = float(
            reading.rms_mm_s
        )

        thresholds = self.thresholds

        if (
            value
            >= thresholds.rms_critical_mm_s
        ):
            return [
                self._anomaly(
                    reading,
                    VibrationAnomalyType.HIGH_RMS,
                    VibrationSeverity.CRITICAL,
                    1.0,
                    [
                        VibrationEvidence(
                            metric="rms_mm_s",
                            value=value,
                            threshold=(
                                thresholds
                                .rms_critical_mm_s
                            ),
                            score=1.0,
                            description=(
                                "Critical vibration RMS "
                                "level detected"
                            ),
                        )
                    ],
                    maintenance=True,
                    safety=True,
                )
            ]

        if (
            value
            >= thresholds.rms_warning_mm_s
        ):
            score = _normalize(
                value,
                thresholds.rms_warning_mm_s,
                thresholds.rms_critical_mm_s,
            )

            return [
                self._anomaly(
                    reading,
                    VibrationAnomalyType.HIGH_RMS,
                    VibrationSeverity.HIGH,
                    max(
                        0.70,
                        score,
                    ),
                    [
                        VibrationEvidence(
                            metric="rms_mm_s",
                            value=value,
                            threshold=(
                                thresholds
                                .rms_warning_mm_s
                            ),
                            score=score,
                            description=(
                                "Vibration RMS exceeds "
                                "configured warning level"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            ]

        return []

    # ========================================================
    # Peak
    # ========================================================

    def _check_peak(
        self,
        reading: VibrationReading,
    ) -> list[VibrationAnomaly]:

        if not _valid_number(
            reading.peak_mm_s
        ):
            return []

        value = float(
            reading.peak_mm_s
        )

        t = self.thresholds

        if (
            value
            >= t.peak_critical_mm_s
        ):
            severity = (
                VibrationSeverity.CRITICAL
            )
            score = 1.0

        elif (
            value
            >= t.peak_warning_mm_s
        ):
            severity = (
                VibrationSeverity.HIGH
            )

            score = max(
                0.70,
                _normalize(
                    value,
                    t.peak_warning_mm_s,
                    t.peak_critical_mm_s,
                ),
            )

        else:
            return []

        return [
            self._anomaly(
                reading,
                VibrationAnomalyType.HIGH_PEAK,
                severity,
                score,
                [
                    VibrationEvidence(
                        metric="peak_mm_s",
                        value=value,
                        threshold=(
                            t.peak_warning_mm_s
                        ),
                        score=score,
                        description=(
                            "High vibration peak "
                            "detected"
                        ),
                    )
                ],
                maintenance=True,
                safety=(
                    severity
                    == VibrationSeverity.CRITICAL
                ),
            )
        ]

    # ========================================================
    # Crest factor
    # ========================================================

    def _check_crest_factor(
        self,
        reading: VibrationReading,
    ) -> list[VibrationAnomaly]:

        value = reading.crest_factor

        # Derive if not explicitly supplied.
        if (
            value is None
            and _valid_number(
                reading.peak_mm_s
            )
            and _valid_number(
                reading.rms_mm_s
            )
            and float(
                reading.rms_mm_s
            ) > 1e-12
        ):
            value = (
                float(reading.peak_mm_s)
                / float(reading.rms_mm_s)
            )

        if not _valid_number(value):
            return []

        value = float(value)

        t = self.thresholds

        if (
            value
            >= t.crest_factor_critical
        ):
            severity = (
                VibrationSeverity.HIGH
            )
            score = 0.90

        elif (
            value
            >= t.crest_factor_warning
        ):
            severity = (
                VibrationSeverity.MEDIUM
            )

            score = max(
                0.50,
                _normalize(
                    value,
                    t.crest_factor_warning,
                    t.crest_factor_critical,
                ),
            )

        else:
            return []

        return [
            self._anomaly(
                reading,
                VibrationAnomalyType.HIGH_CREST_FACTOR,
                severity,
                score,
                [
                    VibrationEvidence(
                        metric="crest_factor",
                        value=value,
                        threshold=(
                            t.crest_factor_warning
                        ),
                        score=score,
                        description=(
                            "Elevated crest factor may "
                            "indicate impulsive mechanical "
                            "events or bearing damage"
                        ),
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Kurtosis
    # ========================================================

    def _check_kurtosis(
        self,
        reading: VibrationReading,
    ) -> list[VibrationAnomaly]:

        if not _valid_number(
            reading.kurtosis
        ):
            return []

        value = float(
            reading.kurtosis
        )

        t = self.thresholds

        if (
            value >= t.kurtosis_critical
        ):
            severity = (
                VibrationSeverity.HIGH
            )
            score = 0.90

        elif (
            value >= t.kurtosis_warning
        ):
            severity = (
                VibrationSeverity.MEDIUM
            )

            score = max(
                0.50,
                _normalize(
                    value,
                    t.kurtosis_warning,
                    t.kurtosis_critical,
                ),
            )

        else:
            return []

        return [
            self._anomaly(
                reading,
                VibrationAnomalyType.HIGH_KURTOSIS,
                severity,
                score,
                [
                    VibrationEvidence(
                        metric="kurtosis",
                        value=value,
                        threshold=(
                            t.kurtosis_warning
                        ),
                        score=score,
                        description=(
                            "Elevated vibration kurtosis "
                            "indicates impulsive behavior"
                        ),
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Rate of change
    # ========================================================

    @staticmethod
    def _rms_rate(
        current: VibrationReading,
        previous: VibrationReading | None,
    ) -> float | None:

        if (
            previous is None
            or not _valid_number(
                current.rms_mm_s
            )
            or not _valid_number(
                previous.rms_mm_s
            )
        ):
            return None

        seconds = (
            current.timestamp
            - previous.timestamp
        ).total_seconds()

        if seconds <= 0:
            return None

        delta = (
            float(current.rms_mm_s)
            - float(previous.rms_mm_s)
        )

        return (
            delta
            / seconds
            * 60.0
        )

    def _check_rate(
        self,
        reading: VibrationReading,
        rate: float | None,
    ) -> list[VibrationAnomaly]:

        if rate is None:
            return []

        t = self.thresholds

        if (
            rate
            >= t.critical_rms_increase_per_minute
        ):
            return [
                self._anomaly(
                    reading,
                    VibrationAnomalyType.RAPID_INCREASE,
                    VibrationSeverity.CRITICAL,
                    1.0,
                    [
                        VibrationEvidence(
                            metric=(
                                "rms_rate_mm_s_per_min"
                            ),
                            value=rate,
                            threshold=(
                                t
                                .critical_rms_increase_per_minute
                            ),
                            score=1.0,
                            description=(
                                "Critical rapid increase "
                                "in vibration"
                            ),
                        )
                    ],
                    maintenance=True,
                    safety=True,
                )
            ]

        if (
            rate
            >= t.rapid_rms_increase_per_minute
        ):
            score = _normalize(
                rate,
                t.rapid_rms_increase_per_minute,
                t.critical_rms_increase_per_minute,
            )

            return [
                self._anomaly(
                    reading,
                    VibrationAnomalyType.RAPID_INCREASE,
                    VibrationSeverity.HIGH,
                    max(0.70, score),
                    [
                        VibrationEvidence(
                            metric=(
                                "rms_rate_mm_s_per_min"
                            ),
                            value=rate,
                            threshold=(
                                t
                                .rapid_rms_increase_per_minute
                            ),
                            score=score,
                            description=(
                                "Rapid vibration increase "
                                "detected"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            ]

        return []

    # ========================================================
    # Statistical anomaly
    # ========================================================

    def _check_statistical_rms(
        self,
        reading: VibrationReading,
        history: VibrationHistory,
    ) -> list[VibrationAnomaly]:

        if not _valid_number(
            reading.rms_mm_s
        ):
            return []

        values = history.rms_values()

        if (
            len(values)
            < self.policy.min_history
        ):
            return []

        mu = mean(values)
        sigma = pstdev(values)

        if sigma <= 1e-12:
            return []

        value = float(
            reading.rms_mm_s
        )

        z = (
            value - mu
        ) / sigma

        absolute_z = abs(z)

        if (
            absolute_z
            < self.policy.z_anomaly
        ):
            return []

        anomaly_type = (
            VibrationAnomalyType.SPIKE
            if z > 0
            else VibrationAnomalyType.DROP
        )

        if (
            absolute_z
            >= self.policy.z_critical
        ):
            severity = (
                VibrationSeverity.HIGH
            )
            score = 0.90
        else:
            severity = (
                VibrationSeverity.MEDIUM
            )
            score = max(
                0.55,
                _normalize(
                    absolute_z,
                    self.policy.z_anomaly,
                    self.policy.z_critical,
                ),
            )

        return [
            self._anomaly(
                reading,
                anomaly_type,
                severity,
                score,
                [
                    VibrationEvidence(
                        metric="rms_z_score",
                        value=z,
                        threshold=(
                            self.policy.z_anomaly
                        ),
                        score=score,
                        description=(
                            "Vibration RMS differs "
                            "significantly from historical "
                            "baseline"
                        ),
                        metadata={
                            "baseline_mean":
                                mu,
                            "baseline_std":
                                sigma,
                        },
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Baseline shift
    # ========================================================

    def _check_baseline_shift(
        self,
        reading: VibrationReading,
        history: VibrationHistory,
    ) -> list[VibrationAnomaly]:

        if not _valid_number(
            reading.rms_mm_s
        ):
            return []

        values = history.rms_values()

        window = (
            self.policy.drift_window
        )

        if len(values) < window * 2:
            return []

        reference = values[:-window]

        recent = (
            values[-window + 1 :]
            + [float(reading.rms_mm_s)]
        )

        if not reference:
            return []

        reference_mean = mean(
            reference
        )

        reference_std = pstdev(
            reference
        )

        if reference_std <= 1e-12:
            return []

        recent_mean = mean(
            recent
        )

        shift_z = (
            recent_mean
            - reference_mean
        ) / reference_std

        if (
            shift_z
            < self.policy.drift_z_threshold
        ):
            return []

        score = _clamp(
            shift_z
            / (
                self.policy
                .drift_z_threshold
                * 2.0
            )
        )

        return [
            self._anomaly(
                reading,
                VibrationAnomalyType.BASELINE_SHIFT,
                VibrationSeverity.MEDIUM,
                max(0.55, score),
                [
                    VibrationEvidence(
                        metric="rms_baseline_shift_z",
                        value=shift_z,
                        threshold=(
                            self.policy
                            .drift_z_threshold
                        ),
                        score=score,
                        description=(
                            "Vibration baseline has "
                            "shifted upward"
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
    # Long-term trend
    # ========================================================

    def _check_trend(
        self,
        reading: VibrationReading,
        history: VibrationHistory,
    ) -> list[VibrationAnomaly]:

        points = (
            history.recent(
                self.policy.trend_window - 1
            )
            + [reading]
        )

        points = [
            point
            for point in points
            if _valid_number(
                point.rms_mm_s
            )
        ]

        if (
            len(points)
            < self.policy.trend_window
        ):
            return []

        slope = _linear_rms_slope(
            points
        )

        threshold = (
            self.thresholds
            .rapid_rms_increase_per_minute
            * 0.20
        )

        if slope < threshold:
            return []

        score = _clamp(
            slope
            / max(
                self.thresholds
                .rapid_rms_increase_per_minute,
                1e-9,
            )
        )

        return [
            self._anomaly(
                reading,
                VibrationAnomalyType.GRADUAL_INCREASE,
                VibrationSeverity.MEDIUM,
                max(0.45, score),
                [
                    VibrationEvidence(
                        metric=(
                            "rms_trend_mm_s_per_min"
                        ),
                        value=slope,
                        threshold=threshold,
                        score=score,
                        description=(
                            "Persistent upward vibration "
                            "trend detected"
                        ),
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Variability
    # ========================================================

    def _check_variability(
        self,
        reading: VibrationReading,
        history: VibrationHistory,
    ) -> list[VibrationAnomaly]:

        points = (
            history.recent(
                self.policy
                .variability_window - 1
            )
            + [reading]
        )

        values = [
            float(point.rms_mm_s)
            for point in points
            if _valid_number(
                point.rms_mm_s
            )
        ]

        if (
            len(values)
            < self.policy.variability_window
        ):
            return []

        mu = mean(values)
        sigma = pstdev(values)

        ratio = (
            sigma
            / max(
                abs(mu),
                1e-6,
            )
        )

        if (
            ratio
            <= self.policy.variability_ratio
        ):
            return []

        score = _clamp(
            ratio
            / max(
                self.policy.variability_ratio
                * 2.0,
                1e-9,
            )
        )

        return [
            self._anomaly(
                reading,
                VibrationAnomalyType.EXCESSIVE_VARIABILITY,
                VibrationSeverity.MEDIUM,
                max(0.50, score),
                [
                    VibrationEvidence(
                        metric=(
                            "rms_variability_ratio"
                        ),
                        value=ratio,
                        threshold=(
                            self.policy
                            .variability_ratio
                        ),
                        score=score,
                        description=(
                            "Vibration signal has "
                            "excessive variability"
                        ),
                    )
                ],
                maintenance=True,
            )
        ]

    # ========================================================
    # Spectral fault indicators
    # ========================================================

    def _check_spectral_faults(
        self,
        reading: VibrationReading,
    ) -> list[VibrationAnomaly]:

        bands = {
            str(key).lower():
                float(value)
            for key, value
            in reading.frequency_bands.items()
            if _valid_number(value)
        }

        if not bands:
            return []

        total_energy = sum(
            max(0.0, value)
            for value in bands.values()
        )

        if total_energy <= 1e-12:
            return []

        anomalies: list[
            VibrationAnomaly
        ] = []

        def ratio(
            *names: str,
        ) -> float:

            energy = sum(
                max(
                    0.0,
                    bands.get(
                        name.lower(),
                        0.0,
                    ),
                )
                for name in names
            )

            return energy / total_energy

        # ----------------------------------------------------
        # Imbalance: dominant 1x rotational component
        # ----------------------------------------------------

        one_x = ratio(
            "1x",
            "1x_rpm",
            "fundamental",
        )

        if (
            one_x
            >= self.policy.one_x_warning_ratio
        ):
            score = _clamp(
                one_x
                / max(
                    self.policy
                    .one_x_warning_ratio
                    * 2.0,
                    1e-9,
                )
            )

            anomalies.append(
                self._anomaly(
                    reading,
                    VibrationAnomalyType.IMBALANCE_SUSPECTED,
                    VibrationSeverity.MEDIUM,
                    max(0.55, score),
                    [
                        VibrationEvidence(
                            metric="1x_energy_ratio",
                            value=one_x,
                            threshold=(
                                self.policy
                                .one_x_warning_ratio
                            ),
                            score=score,
                            description=(
                                "Strong 1x rotational "
                                "component may indicate "
                                "mechanical imbalance"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            )

        # ----------------------------------------------------
        # Misalignment: 2x component
        # ----------------------------------------------------

        two_x = ratio(
            "2x",
            "2x_rpm",
            "second_harmonic",
        )

        if (
            two_x
            >= self.policy.two_x_warning_ratio
        ):
            score = _clamp(
                two_x
                / max(
                    self.policy
                    .two_x_warning_ratio
                    * 2.0,
                    1e-9,
                )
            )

            anomalies.append(
                self._anomaly(
                    reading,
                    VibrationAnomalyType.MISALIGNMENT_SUSPECTED,
                    VibrationSeverity.MEDIUM,
                    max(0.55, score),
                    [
                        VibrationEvidence(
                            metric="2x_energy_ratio",
                            value=two_x,
                            threshold=(
                                self.policy
                                .two_x_warning_ratio
                            ),
                            score=score,
                            description=(
                                "Elevated 2x rotational "
                                "component may indicate "
                                "shaft misalignment"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            )

        # ----------------------------------------------------
        # Looseness: multiple rotational harmonics
        # ----------------------------------------------------

        harmonic_ratio = ratio(
            "2x",
            "2x_rpm",
            "3x",
            "3x_rpm",
            "4x",
            "4x_rpm",
        )

        if harmonic_ratio >= 0.45:
            anomalies.append(
                self._anomaly(
                    reading,
                    VibrationAnomalyType.LOOSENESS_SUSPECTED,
                    VibrationSeverity.MEDIUM,
                    _clamp(harmonic_ratio),
                    [
                        VibrationEvidence(
                            metric=(
                                "harmonic_energy_ratio"
                            ),
                            value=harmonic_ratio,
                            threshold=0.45,
                            score=_clamp(
                                harmonic_ratio
                            ),
                            description=(
                                "Multiple rotational "
                                "harmonics may indicate "
                                "mechanical looseness"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            )

        # ----------------------------------------------------
        # Bearing fault bands
        # ----------------------------------------------------

        bearing_ratio = ratio(
            "bearing",
            "bpfo",
            "bpfi",
            "bsf",
            "ftf",
            "bearing_high_frequency",
        )

        if (
            bearing_ratio
            >= self.policy
            .bearing_band_warning_ratio
        ):
            critical = (
                bearing_ratio
                >= self.policy
                .bearing_band_critical_ratio
            )

            severity = (
                VibrationSeverity.HIGH
                if critical
                else VibrationSeverity.MEDIUM
            )

            score = _normalize(
                bearing_ratio,
                self.policy
                .bearing_band_warning_ratio,
                self.policy
                .bearing_band_critical_ratio,
            )

            anomalies.append(
                self._anomaly(
                    reading,
                    VibrationAnomalyType.BEARING_FAULT_SUSPECTED,
                    severity,
                    max(
                        0.60,
                        score,
                    ),
                    [
                        VibrationEvidence(
                            metric=(
                                "bearing_band_energy_ratio"
                            ),
                            value=bearing_ratio,
                            threshold=(
                                self.policy
                                .bearing_band_warning_ratio
                            ),
                            score=score,
                            description=(
                                "Elevated bearing-related "
                                "spectral energy detected"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            )

        # ----------------------------------------------------
        # Gear fault bands
        # ----------------------------------------------------

        gear_ratio = ratio(
            "gear",
            "gear_mesh",
            "gear_mesh_frequency",
            "gmf",
        )

        if (
            gear_ratio
            >= self.policy
            .gear_band_warning_ratio
        ):
            critical = (
                gear_ratio
                >= self.policy
                .gear_band_critical_ratio
            )

            severity = (
                VibrationSeverity.HIGH
                if critical
                else VibrationSeverity.MEDIUM
            )

            score = _normalize(
                gear_ratio,
                self.policy
                .gear_band_warning_ratio,
                self.policy
                .gear_band_critical_ratio,
            )

            anomalies.append(
                self._anomaly(
                    reading,
                    VibrationAnomalyType.GEAR_FAULT_SUSPECTED,
                    severity,
                    max(
                        0.60,
                        score,
                    ),
                    [
                        VibrationEvidence(
                            metric=(
                                "gear_band_energy_ratio"
                            ),
                            value=gear_ratio,
                            threshold=(
                                self.policy
                                .gear_band_warning_ratio
                            ),
                            score=score,
                            description=(
                                "Elevated gear-mesh "
                                "spectral energy detected"
                            ),
                        )
                    ],
                    maintenance=True,
                )
            )

        return anomalies

    # ========================================================
    # Multi-indicator correlation
    # ========================================================

    def _apply_multi_indicator(
        self,
        reading: VibrationReading,
        anomalies: list[VibrationAnomaly],
    ) -> list[VibrationAnomaly]:

        meaningful = [
            anomaly
            for anomaly in anomalies
            if anomaly.anomaly_type
            != VibrationAnomalyType.MULTI_INDICATOR
        ]

        if (
            len(meaningful)
            < self.policy.multi_indicator_count
        ):
            return anomalies

        unique_types = {
            anomaly.anomaly_type
            for anomaly in meaningful
        }

        if (
            len(unique_types)
            < self.policy.multi_indicator_count
        ):
            return anomalies

        strongest = max(
            anomaly.score
            for anomaly in meaningful
        )

        score = _clamp(
            strongest + 0.10
        )

        severity = (
            VibrationSeverity.HIGH
            if score >= 0.70
            else VibrationSeverity.MEDIUM
        )

        anomalies.append(
            self._anomaly(
                reading,
                VibrationAnomalyType.MULTI_INDICATOR,
                severity,
                score,
                [
                    VibrationEvidence(
                        metric=(
                            "correlated_vibration_indicators"
                        ),
                        value=len(unique_types),
                        threshold=(
                            self.policy
                            .multi_indicator_count
                        ),
                        score=score,
                        description=(
                            "Multiple independent "
                            "vibration indicators are "
                            "simultaneously abnormal"
                        ),
                        metadata={
                            "indicators": [
                                item.value
                                for item in unique_types
                            ]
                        },
                    )
                ],
                maintenance=True,
            )
        )

        return anomalies

    # ========================================================
    # Baseline information
    # ========================================================

    def _baseline_statistics(
        self,
        reading: VibrationReading,
        history: VibrationHistory,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
    ]:

        values = history.rms_values()

        if not values:
            return None, None, None

        mu = mean(values)

        sigma = (
            pstdev(values)
            if len(values) > 1
            else 0.0
        )

        if (
            sigma <= 1e-12
            or not _valid_number(
                reading.rms_mm_s
            )
        ):
            return (
                mu,
                sigma,
                None,
            )

        z = (
            float(reading.rms_mm_s)
            - mu
        ) / sigma

        return (
            mu,
            sigma,
            z,
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _history_for(
        self,
        reading: VibrationReading,
    ) -> VibrationHistory:

        key = (
            reading.sensor_id
            or reading.component_id
            or reading.asset_id
            or "default"
        )

        history = self._histories.get(
            key
        )

        if history is None:
            history = VibrationHistory(
                self.policy.max_history
            )

            self._histories[key] = (
                history
            )

        return history

    @staticmethod
    def _has_measurement(
        reading: VibrationReading,
    ) -> bool:

        return any(
            _valid_number(value)
            for value in (
                reading.rms_mm_s,
                reading.peak_mm_s,
                reading.acceleration_rms_g,
                reading.acceleration_peak_g,
                reading.crest_factor,
                reading.kurtosis,
            )
        ) or bool(
            reading.frequency_bands
        )

    def _anomaly(
        self,
        reading: VibrationReading,
        anomaly_type: VibrationAnomalyType,
        severity: VibrationSeverity,
        score: float,
        evidence: list[VibrationEvidence],
        *,
        maintenance: bool = False,
        safety: bool = False,
    ) -> VibrationAnomaly:

        return VibrationAnomaly(
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
            requires_maintenance_review=(
                maintenance
            ),
            requires_safety_review=safety,
            metadata={
                "axis": reading.axis,
                "rpm": reading.rpm,
                "load_percent":
                    reading.load_percent,
                "site_id":
                    reading.site_id,
                "zone_id":
                    reading.zone_id,
            },
        )

    def _overall_severity(
        self,
        anomalies: Sequence[
            VibrationAnomaly
        ],
    ) -> VibrationSeverity:

        if not anomalies:
            return (
                VibrationSeverity.NORMAL
            )

        return max(
            (
                anomaly.severity
                for anomaly in anomalies
            ),
            key=lambda severity:
                self._SEVERITY_RANK[
                    severity
                ],
        )

    @staticmethod
    def _overall_score(
        anomalies: Sequence[
            VibrationAnomaly
        ],
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
                len(scores) - 1,
            ) * 0.04,
        )

        return _clamp(
            strongest
            + corroboration_bonus
        )

    @staticmethod
    def _trend_state(
        rate: float | None,
    ) -> VibrationTrend:

        if rate is None:
            return (
                VibrationTrend.UNKNOWN
            )

        if rate >= 1.5:
            return (
                VibrationTrend.RAPIDLY_RISING
            )

        if rate > 0.05:
            return VibrationTrend.RISING

        if rate < -0.05:
            return VibrationTrend.FALLING

        return VibrationTrend.STABLE

    @staticmethod
    def _state(
        severity: VibrationSeverity,
    ) -> VibrationState:

        mapping = {
            VibrationSeverity.NORMAL:
                VibrationState.NORMAL,

            VibrationSeverity.LOW:
                VibrationState.ELEVATED,

            VibrationSeverity.MEDIUM:
                VibrationState.ANOMALOUS,

            VibrationSeverity.HIGH:
                VibrationState.ANOMALOUS,

            VibrationSeverity.CRITICAL:
                VibrationState.CRITICAL,
        }

        return mapping[severity]

    def _deduplicate(
        self,
        anomalies: Sequence[
            VibrationAnomaly
        ],
    ) -> list[VibrationAnomaly]:

        selected: dict[
            VibrationAnomalyType,
            VibrationAnomaly,
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

            old_rank = (
                self._SEVERITY_RANK[
                    existing.severity
                ]
            )

            new_rank = (
                self._SEVERITY_RANK[
                    anomaly.severity
                ]
            )

            if (
                new_rank > old_rank
                or (
                    new_rank == old_rank
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

    @staticmethod
    def _suspected_faults(
        anomalies: Sequence[
            VibrationAnomaly
        ],
    ) -> list[str]:

        mapping = {
            VibrationAnomalyType.IMBALANCE_SUSPECTED:
                "rotor_imbalance",

            VibrationAnomalyType.MISALIGNMENT_SUSPECTED:
                "shaft_misalignment",

            VibrationAnomalyType.LOOSENESS_SUSPECTED:
                "mechanical_looseness",

            VibrationAnomalyType.BEARING_FAULT_SUSPECTED:
                "bearing_fault",

            VibrationAnomalyType.GEAR_FAULT_SUSPECTED:
                "gear_fault",

            VibrationAnomalyType.RESONANCE_SUSPECTED:
                "resonance",
        }

        return sorted(
            {
                mapping[
                    anomaly.anomaly_type
                ]
                for anomaly in anomalies
                if anomaly.anomaly_type
                in mapping
            }
        )

    def _insufficient(
        self,
        reading: VibrationReading,
        history: VibrationHistory,
        reason: str,
    ) -> VibrationAssessment:

        return VibrationAssessment(
            assessment_id=str(uuid4()),
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            component_id=reading.component_id,
            state=(
                VibrationState
                .INSUFFICIENT_DATA
            ),
            severity=(
                VibrationSeverity.NORMAL
            ),
            anomaly_score=0.0,
            confidence=reading.confidence,
            rms_mm_s=reading.rms_mm_s,
            rms_rate_per_minute=None,
            trend=VibrationTrend.UNKNOWN,
            baseline_mean_rms=None,
            baseline_std_rms=None,
            rms_z_score=None,
            anomalies=[],
            evidence=[
                VibrationEvidence(
                    metric="vibration",
                    value=reading.rms_mm_s,
                    score=0.0,
                    description=reason,
                )
            ],
            suspected_faults=[],
            metadata={
                "history_size":
                    history.size,
            },
        )


# ============================================================
# Trend calculation
# ============================================================


def _linear_rms_slope(
    readings: Sequence[
        VibrationReading
    ],
) -> float:
    """
    Least-squares RMS trend in mm/s per minute.
    """

    if len(readings) < 2:
        return 0.0

    start = readings[0].timestamp

    x: list[float] = []
    y: list[float] = []

    for reading in readings:

        if not _valid_number(
            reading.rms_mm_s
        ):
            continue

        minutes = (
            reading.timestamp - start
        ).total_seconds() / 60.0

        x.append(minutes)

        y.append(
            float(reading.rms_mm_s)
        )

    if len(x) < 2:
        return 0.0

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum(
        (xi - mean_x)
        * (yi - mean_y)
        for xi, yi
        in zip(x, y)
    )

    denominator = sum(
        (xi - mean_x) ** 2
        for xi in x
    )

    if denominator <= 1e-12:
        return 0.0

    return numerator / denominator


# ============================================================
# Convenience constructor
# ============================================================


def create_vibration_reading(
    *,
    rms_mm_s: float | None = None,
    peak_mm_s: float | None = None,
    acceleration_rms_g: float | None = None,
    acceleration_peak_g: float | None = None,
    crest_factor: float | None = None,
    kurtosis: float | None = None,
    rpm: float | None = None,
    load_percent: float | None = None,
    frequency_bands: Mapping[
        str,
        float,
    ] | None = None,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    axis: str | None = None,
    confidence: float = 1.0,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> VibrationReading:

    return VibrationReading(
        timestamp=(
            timestamp
            or datetime.now(timezone.utc)
        ),
        rms_mm_s=rms_mm_s,
        peak_mm_s=peak_mm_s,
        acceleration_rms_g=(
            acceleration_rms_g
        ),
        acceleration_peak_g=(
            acceleration_peak_g
        ),
        crest_factor=crest_factor,
        kurtosis=kurtosis,
        rpm=rpm,
        load_percent=load_percent,
        frequency_bands=(
            frequency_bands or {}
        ),
        asset_id=asset_id,
        sensor_id=sensor_id,
        component_id=component_id,
        axis=axis,
        confidence=confidence,
        metadata=metadata or {},
    )


# ============================================================
# Utility functions
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
    except (
        TypeError,
        ValueError,
    ):
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


default_vibration_anomaly_detector = (
    VibrationAnomalyDetector()
)


__all__ = [
    "VibrationAnomalyType",
    "VibrationSeverity",
    "VibrationState",
    "VibrationTrend",
    "FrequencyBand",
    "VibrationReading",
    "VibrationThresholds",
    "VibrationPolicy",
    "VibrationEvidence",
    "VibrationAnomaly",
    "VibrationAssessment",
    "VibrationHistory",
    "VibrationAnomalyDetector",
    "create_vibration_reading",
    "default_vibration_anomaly_detector",
]