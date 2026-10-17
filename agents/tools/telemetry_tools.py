    # app/agents/tools/telemetry_tools.py

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timedelta, timezone
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


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


def finite(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def ensure_utc(
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(timezone.utc)


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
        return [
            json_safe(item)
            for item in value
        ]

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


# =========================================================
# Enums
# =========================================================

class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    ENERGY = "energy"
    SPEED = "speed"
    RPM = "rpm"
    FLOW = "flow"
    HUMIDITY = "humidity"
    TORQUE = "torque"
    SOUND = "sound"
    OTHER = "other"


class TelemetryQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    BAD = "bad"
    MISSING = "missing"
    STALE = "stale"
    OUT_OF_RANGE = "out_of_range"
    INVALID = "invalid"


class TelemetryStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class TrendDirection(str, Enum):
    STRONGLY_DECREASING = "strongly_decreasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"
    STRONGLY_INCREASING = "strongly_increasing"
    UNKNOWN = "unknown"


class AnomalyLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class TelemetryEvidenceType(str, Enum):
    HIGH_TEMPERATURE = "high_temperature"
    HIGH_VIBRATION = "high_vibration"
    PRESSURE_ANOMALY = "pressure_anomaly"
    CURRENT_ANOMALY = "current_anomaly"
    VOLTAGE_ANOMALY = "voltage_anomaly"
    ENERGY_ANOMALY = "energy_anomaly"
    RAPID_CHANGE = "rapid_change"
    SENSOR_FAILURE = "sensor_failure"
    STALE_DATA = "stale_data"
    DATA_QUALITY = "data_quality"
    ANOMALY = "anomaly"
    TREND = "trend"


# =========================================================
# Sensor threshold
# =========================================================

@dataclass(slots=True)
class SensorThreshold:
    warning_low: float | None = None
    critical_low: float | None = None

    warning_high: float | None = None
    critical_high: float | None = None

    minimum_valid: float | None = None
    maximum_valid: float | None = None

    maximum_rate_of_change: float | None = None

    unit: str | None = None


# =========================================================
# Telemetry point
# =========================================================

@dataclass(slots=True)
class TelemetryPoint:
    sensor_id: str

    asset_id: str

    sensor_type: SensorType

    value: float

    timestamp: datetime

    unit: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    component: str | None = None

    sequence_number: int | None = None

    quality: TelemetryQuality = (
        TelemetryQuality.GOOD
    )

    source: str = "sensor"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.timestamp = (
            ensure_utc(self.timestamp)
            or utc_now()
        )

        if not math.isfinite(
            finite(
                self.value,
                default=float("nan"),
            )
        ):
            self.quality = (
                TelemetryQuality.INVALID
            )


# =========================================================
# Validation result
# =========================================================

@dataclass(slots=True)
class TelemetryValidationResult:
    valid: bool

    quality: TelemetryQuality

    status: TelemetryStatus

    value: float | None

    reasons: list[str]

    stale: bool = False

    out_of_range: bool = False

    rate_of_change_violation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Statistics
# =========================================================

@dataclass(slots=True)
class TelemetryStatistics:
    count: int

    minimum: float | None
    maximum: float | None

    mean: float | None
    median: float | None

    standard_deviation: float | None

    variance: float | None

    first: float | None
    last: float | None

    delta: float | None

    rate_of_change: float | None

    coefficient_of_variation: float | None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Trend
# =========================================================

@dataclass(slots=True)
class TelemetryTrend:
    direction: TrendDirection

    slope_per_hour: float

    normalized_slope: float

    confidence: float

    sample_count: int

    start_time: datetime | None
    end_time: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Anomaly
# =========================================================

@dataclass(slots=True)
class TelemetryAnomaly:
    anomaly_id: str

    asset_id: str

    sensor_id: str

    sensor_type: SensorType

    score: float

    level: AnomalyLevel

    value: float

    expected_value: float | None

    z_score: float | None

    reason: str

    timestamp: datetime

    component: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Evidence
# =========================================================

@dataclass(slots=True)
class TelemetryEvidence:
    evidence_id: str

    evidence_type: TelemetryEvidenceType

    asset_id: str

    description: str

    severity: float
    confidence: float

    sensor_type: SensorType | None = None

    sensor_id: str | None = None
    component: str | None = None

    value: float | None = None
    unit: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.severity = clamp(
            self.severity
        )

        self.confidence = clamp(
            self.confidence
        )

        self.timestamp = (
            ensure_utc(self.timestamp)
            or utc_now()
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Data quality
# =========================================================

@dataclass(slots=True)
class TelemetryQualityReport:
    asset_id: str

    total_points: int

    valid_points: int
    invalid_points: int

    stale_points: int
    out_of_range_points: int

    missing_sequence_count: int

    duplicate_count: int

    completeness: float
    validity: float
    freshness: float

    overall_quality_score: float

    quality: TelemetryQuality

    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Asset telemetry snapshot
# =========================================================

@dataclass(slots=True)
class AssetTelemetrySnapshot:
    asset_id: str

    timestamp: datetime

    temperature: float | None = None
    vibration: float | None = None
    pressure: float | None = None

    current: float | None = None
    voltage: float | None = None

    power: float | None = None
    energy: float | None = None

    speed: float | None = None
    rpm: float | None = None

    flow: float | None = None

    status: TelemetryStatus = (
        TelemetryStatus.UNKNOWN
    )

    quality_score: float = 0.0

    anomaly_score: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Analysis
# =========================================================

@dataclass(slots=True)
class TelemetryAnalysisResult:
    asset_id: str

    sensor_type: SensorType

    latest: TelemetryPoint | None

    validation: TelemetryValidationResult | None

    statistics: TelemetryStatistics

    trend: TelemetryTrend

    anomalies: list[TelemetryAnomaly]

    evidence: list[TelemetryEvidence]

    status: TelemetryStatus

    confidence: float

    generated_at: datetime = field(
        default_factory=utc_now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class TelemetryToolsConfig:
    stale_after_seconds: float = 120.0

    severe_stale_after_seconds: float = 600.0

    minimum_statistics_samples: int = 3

    minimum_trend_samples: int = 4

    anomaly_z_warning: float = 2.5
    anomaly_z_high: float = 3.0
    anomaly_z_critical: float = 4.0

    trend_normalized_warning: float = 0.10
    trend_normalized_strong: float = 0.25

    duplicate_time_tolerance_seconds: float = 0.001

    sequence_gap_warning: int = 1

    maximum_history_points: int = 10_000


# =========================================================
# Telemetry Tools
# =========================================================

class TelemetryTools:

    def __init__(
        self,
        *,
        config: TelemetryToolsConfig | None = None,
    ) -> None:
        self.config = (
            config
            or TelemetryToolsConfig()
        )

    # =====================================================
    # Validate point
    # =====================================================

    def validate_point(
        self,
        point: TelemetryPoint,
        *,
        threshold: SensorThreshold | None = None,
        previous: TelemetryPoint | None = None,
        now: datetime | None = None,
    ) -> TelemetryValidationResult:

        now = (
            ensure_utc(now)
            or utc_now()
        )

        reasons: list[str] = []

        value = finite(
            point.value,
            default=float("nan"),
        )

        if not math.isfinite(value):
            return TelemetryValidationResult(
                valid=False,
                quality=TelemetryQuality.INVALID,
                status=TelemetryStatus.UNKNOWN,
                value=None,
                reasons=[
                    "Telemetry value is not finite."
                ],
            )

        age_seconds = max(
            0.0,
            (
                now - point.timestamp
            ).total_seconds(),
        )

        stale = (
            age_seconds
            > self.config.stale_after_seconds
        )

        if stale:
            reasons.append(
                f"Telemetry is stale "
                f"({age_seconds:.1f}s old)."
            )

        out_of_range = False
        rate_violation = False

        status = TelemetryStatus.NORMAL
        quality = point.quality

        if threshold is not None:
            if (
                threshold.minimum_valid
                is not None
                and value
                < threshold.minimum_valid
            ):
                out_of_range = True

            if (
                threshold.maximum_valid
                is not None
                and value
                > threshold.maximum_valid
            ):
                out_of_range = True

            if out_of_range:
                reasons.append(
                    "Telemetry value is outside "
                    "the physically valid range."
                )

            if (
                threshold.critical_high
                is not None
                and value
                >= threshold.critical_high
            ):
                status = TelemetryStatus.CRITICAL

            elif (
                threshold.critical_low
                is not None
                and value
                <= threshold.critical_low
            ):
                status = TelemetryStatus.CRITICAL

            elif (
                threshold.warning_high
                is not None
                and value
                >= threshold.warning_high
            ):
                status = TelemetryStatus.WARNING

            elif (
                threshold.warning_low
                is not None
                and value
                <= threshold.warning_low
            ):
                status = TelemetryStatus.WARNING

            if (
                previous is not None
                and threshold.maximum_rate_of_change
                is not None
            ):
                dt = (
                    point.timestamp
                    - previous.timestamp
                ).total_seconds()

                if dt > 0:
                    rate = abs(
                        value - previous.value
                    ) / dt

                    if (
                        rate
                        > threshold
                        .maximum_rate_of_change
                    ):
                        rate_violation = True

                        reasons.append(
                            "Maximum allowed rate of "
                            "change was exceeded."
                        )

                        if (
                            status
                            == TelemetryStatus.NORMAL
                        ):
                            status = (
                                TelemetryStatus.WARNING
                            )

        if out_of_range:
            quality = (
                TelemetryQuality.OUT_OF_RANGE
            )

        elif stale:
            quality = TelemetryQuality.STALE

        elif quality == TelemetryQuality.GOOD:
            quality = TelemetryQuality.GOOD

        valid = quality not in {
            TelemetryQuality.INVALID,
            TelemetryQuality.BAD,
            TelemetryQuality.MISSING,
            TelemetryQuality.OUT_OF_RANGE,
        }

        return TelemetryValidationResult(
            valid=valid,

            quality=quality,

            status=status,

            value=value,

            reasons=reasons,

            stale=stale,

            out_of_range=out_of_range,

            rate_of_change_violation=(
                rate_violation
            ),
        )

    # =====================================================
    # Statistics
    # =====================================================

    def statistics(
        self,
        points: Sequence[TelemetryPoint],
    ) -> TelemetryStatistics:

        ordered = self._ordered_valid_points(
            points
        )

        values = [
            point.value
            for point in ordered
        ]

        if not values:
            return TelemetryStatistics(
                count=0,

                minimum=None,
                maximum=None,

                mean=None,
                median=None,

                standard_deviation=None,
                variance=None,

                first=None,
                last=None,

                delta=None,
                rate_of_change=None,

                coefficient_of_variation=None,
            )

        count = len(values)

        mean = statistics.fmean(values)

        median = statistics.median(values)

        minimum = min(values)
        maximum = max(values)

        if count >= 2:
            variance = statistics.variance(
                values
            )

            std = math.sqrt(variance)

            delta = (
                values[-1]
                - values[0]
            )

            seconds = (
                ordered[-1].timestamp
                - ordered[0].timestamp
            ).total_seconds()

            rate = (
                delta / seconds
                if seconds > 0
                else None
            )

        else:
            variance = 0.0
            std = 0.0
            delta = 0.0
            rate = None

        coefficient = (
            abs(std / mean)
            if abs(mean) > 1e-12
            else None
        )

        return TelemetryStatistics(
            count=count,

            minimum=round(minimum, 6),
            maximum=round(maximum, 6),

            mean=round(mean, 6),
            median=round(median, 6),

            standard_deviation=round(
                std,
                6,
            ),

            variance=round(
                variance,
                6,
            ),

            first=round(
                values[0],
                6,
            ),

            last=round(
                values[-1],
                6,
            ),

            delta=round(
                delta,
                6,
            ),

            rate_of_change=(
                round(rate, 9)
                if rate is not None
                else None
            ),

            coefficient_of_variation=(
                round(coefficient, 6)
                if coefficient is not None
                else None
            ),
        )

    # =====================================================
    # Trend
    # =====================================================

    def trend(
        self,
        points: Sequence[TelemetryPoint],
    ) -> TelemetryTrend:

        ordered = self._ordered_valid_points(
            points
        )

        if (
            len(ordered)
            < self.config.minimum_trend_samples
        ):
            return TelemetryTrend(
                direction=TrendDirection.UNKNOWN,

                slope_per_hour=0.0,
                normalized_slope=0.0,

                confidence=0.0,

                sample_count=len(ordered),

                start_time=(
                    ordered[0].timestamp
                    if ordered
                    else None
                ),

                end_time=(
                    ordered[-1].timestamp
                    if ordered
                    else None
                ),
            )

        start = ordered[0].timestamp

        x = [
            (
                point.timestamp - start
            ).total_seconds()
            / 3600.0
            for point in ordered
        ]

        y = [
            point.value
            for point in ordered
        ]

        x_mean = statistics.fmean(x)
        y_mean = statistics.fmean(y)

        numerator = sum(
            (xi - x_mean)
            * (yi - y_mean)
            for xi, yi in zip(x, y)
        )

        denominator = sum(
            (xi - x_mean) ** 2
            for xi in x
        )

        slope = (
            numerator / denominator
            if denominator > 0
            else 0.0
        )

        scale = max(
            abs(y_mean),
            statistics.pstdev(y)
            if len(y) > 1
            else 0.0,
            1e-9,
        )

        normalized = slope / scale

        magnitude = abs(normalized)

        if (
            magnitude
            < self.config
            .trend_normalized_warning
        ):
            direction = TrendDirection.STABLE

        elif normalized >= (
            self.config
            .trend_normalized_strong
        ):
            direction = (
                TrendDirection
                .STRONGLY_INCREASING
            )

        elif normalized > 0:
            direction = (
                TrendDirection.INCREASING
            )

        elif normalized <= -(
            self.config
            .trend_normalized_strong
        ):
            direction = (
                TrendDirection
                .STRONGLY_DECREASING
            )

        else:
            direction = (
                TrendDirection.DECREASING
            )

        predicted = [
            y_mean
            + slope * (xi - x_mean)
            for xi in x
        ]

        ss_res = sum(
            (yi - pi) ** 2
            for yi, pi in zip(
                y,
                predicted,
            )
        )

        ss_tot = sum(
            (yi - y_mean) ** 2
            for yi in y
        )

        if ss_tot <= 1e-12:
            r_squared = (
                1.0
                if ss_res <= 1e-12
                else 0.0
            )
        else:
            r_squared = clamp(
                1.0 - ss_res / ss_tot
            )

        sample_confidence = clamp(
            len(ordered) / 20.0
        )

        confidence = (
            0.65 * r_squared
            + 0.35 * sample_confidence
        )

        return TelemetryTrend(
            direction=direction,

            slope_per_hour=round(
                slope,
                8,
            ),

            normalized_slope=round(
                normalized,
                8,
            ),

            confidence=round(
                clamp(confidence),
                4,
            ),

            sample_count=len(ordered),

            start_time=ordered[0].timestamp,
            end_time=ordered[-1].timestamp,
        )

    # =====================================================
    # Statistical anomaly
    # =====================================================

    def detect_anomalies(
        self,
        points: Sequence[TelemetryPoint],
        *,
        threshold: SensorThreshold | None = None,
    ) -> list[TelemetryAnomaly]:

        ordered = self._ordered_valid_points(
            points
        )

        if len(ordered) < 3:
            return []

        values = [
            point.value
            for point in ordered
        ]

        median = statistics.median(values)

        absolute_deviations = [
            abs(value - median)
            for value in values
        ]

        mad = statistics.median(
            absolute_deviations
        )

        if mad <= 1e-12:
            mean = statistics.fmean(values)

            std = statistics.pstdev(values)

            if std <= 1e-12:
                return []

            def z_score(value: float) -> float:
                return (
                    value - mean
                ) / std

            expected = mean

        else:
            # Robust z-score using MAD.
            def z_score(value: float) -> float:
                return (
                    0.6745
                    * (value - median)
                    / mad
                )

            expected = median

        anomalies: list[
            TelemetryAnomaly
        ] = []

        for point in ordered:
            z = z_score(point.value)

            score = clamp(
                abs(z)
                / self.config
                .anomaly_z_critical
            )

            if (
                abs(z)
                >= self.config
                .anomaly_z_critical
            ):
                level = AnomalyLevel.CRITICAL

            elif (
                abs(z)
                >= self.config
                .anomaly_z_high
            ):
                level = AnomalyLevel.HIGH

            elif (
                abs(z)
                >= self.config
                .anomaly_z_warning
            ):
                level = AnomalyLevel.MODERATE

            else:
                continue

            reason = (
                f"Telemetry deviates from its "
                f"recent baseline "
                f"(robust z={z:.2f})."
            )

            if threshold is not None:
                validation = self.validate_point(
                    point,
                    threshold=threshold,
                    now=point.timestamp,
                )

                if (
                    validation.status
                    == TelemetryStatus.CRITICAL
                ):
                    level = AnomalyLevel.CRITICAL
                    score = max(score, 0.95)

            anomalies.append(
                TelemetryAnomaly(
                    anomaly_id=new_id(),

                    asset_id=point.asset_id,

                    sensor_id=point.sensor_id,

                    sensor_type=(
                        point.sensor_type
                    ),

                    score=round(
                        score,
                        4,
                    ),

                    level=level,

                    value=point.value,

                    expected_value=round(
                        expected,
                        6,
                    ),

                    z_score=round(
                        z,
                        4,
                    ),

                    reason=reason,

                    timestamp=point.timestamp,

                    component=point.component,
                )
            )

        return anomalies

    # =====================================================
    # Data quality
    # =====================================================

    def quality_report(
        self,
        points: Sequence[TelemetryPoint],
        *,
        asset_id: str,
        expected_points: int | None = None,
        now: datetime | None = None,
    ) -> TelemetryQualityReport:

        now = (
            ensure_utc(now)
            or utc_now()
        )

        total = len(points)

        valid = 0
        invalid = 0
        stale = 0
        out_of_range = 0

        duplicate_count = 0

        ordered = sorted(
            points,
            key=lambda point: (
                point.timestamp,
                point.sensor_id,
            ),
        )

        seen: set[
            tuple[str, datetime, float]
        ] = set()

        for point in ordered:
            key = (
                point.sensor_id,
                point.timestamp,
                point.value,
            )

            if key in seen:
                duplicate_count += 1
            else:
                seen.add(key)

            if point.quality in {
                TelemetryQuality.INVALID,
                TelemetryQuality.BAD,
                TelemetryQuality.MISSING,
            }:
                invalid += 1
            else:
                valid += 1

            if point.quality == (
                TelemetryQuality.OUT_OF_RANGE
            ):
                out_of_range += 1

            age = (
                now - point.timestamp
            ).total_seconds()

            if (
                age
                > self.config.stale_after_seconds
            ):
                stale += 1

        sequence_values = sorted(
            {
                point.sequence_number
                for point in ordered
                if point.sequence_number
                is not None
            }
        )

        missing_sequences = 0

        if len(sequence_values) >= 2:
            expected_sequence_count = (
                sequence_values[-1]
                - sequence_values[0]
                + 1
            )

            missing_sequences = max(
                0,
                expected_sequence_count
                - len(sequence_values),
            )

        denominator = max(
            expected_points or total,
            1,
        )

        completeness = clamp(
            total / denominator
        )

        validity = clamp(
            valid / max(total, 1)
        )

        freshness = clamp(
            1.0
            - stale / max(total, 1)
        )

        duplicate_score = clamp(
            1.0
            - duplicate_count
            / max(total, 1)
        )

        overall = clamp(
            0.30 * completeness
            + 0.30 * validity
            + 0.25 * freshness
            + 0.15 * duplicate_score
        )

        warnings: list[str] = []

        if completeness < 0.80:
            warnings.append(
                "Telemetry completeness is low."
            )

        if validity < 0.90:
            warnings.append(
                "Telemetry contains invalid data."
            )

        if freshness < 0.90:
            warnings.append(
                "Telemetry contains stale data."
            )

        if missing_sequences:
            warnings.append(
                f"{missing_sequences} sequence "
                "value(s) are missing."
            )

        if duplicate_count:
            warnings.append(
                f"{duplicate_count} duplicate "
                "telemetry point(s) detected."
            )

        if overall >= 0.90:
            quality = TelemetryQuality.GOOD

        elif overall >= 0.70:
            quality = TelemetryQuality.DEGRADED

        else:
            quality = TelemetryQuality.BAD

        return TelemetryQualityReport(
            asset_id=asset_id,

            total_points=total,

            valid_points=valid,
            invalid_points=invalid,

            stale_points=stale,

            out_of_range_points=(
                out_of_range
            ),

            missing_sequence_count=(
                missing_sequences
            ),

            duplicate_count=duplicate_count,

            completeness=round(
                completeness,
                4,
            ),

            validity=round(
                validity,
                4,
            ),

            freshness=round(
                freshness,
                4,
            ),

            overall_quality_score=round(
                overall,
                4,
            ),

            quality=quality,

            warnings=warnings,
        )

    # =====================================================
    # Sensor evidence
    # =====================================================

    def build_evidence(
        self,
        *,
        point: TelemetryPoint,
        validation: TelemetryValidationResult,
        trend: TelemetryTrend | None = None,
        anomaly: TelemetryAnomaly | None = None,
    ) -> list[TelemetryEvidence]:

        evidence: list[
            TelemetryEvidence
        ] = []

        if validation.stale:
            evidence.append(
                TelemetryEvidence(
                    evidence_id=new_id(),

                    evidence_type=(
                        TelemetryEvidenceType
                        .STALE_DATA
                    ),

                    asset_id=point.asset_id,

                    description=(
                        f"{point.sensor_type.value} "
                        "telemetry is stale."
                    ),

                    severity=0.45,
                    confidence=1.0,

                    sensor_type=(
                        point.sensor_type
                    ),

                    sensor_id=point.sensor_id,
                    component=point.component,

                    value=point.value,
                    unit=point.unit,

                    timestamp=point.timestamp,
                )
            )

        if not validation.valid:
            evidence.append(
                TelemetryEvidence(
                    evidence_id=new_id(),

                    evidence_type=(
                        TelemetryEvidenceType
                        .SENSOR_FAILURE
                    ),

                    asset_id=point.asset_id,

                    description=(
                        f"{point.sensor_type.value} "
                        "sensor data is invalid."
                    ),

                    severity=0.70,
                    confidence=0.95,

                    sensor_type=(
                        point.sensor_type
                    ),

                    sensor_id=point.sensor_id,
                    component=point.component,

                    value=point.value,
                    unit=point.unit,

                    timestamp=point.timestamp,
                )
            )

        if (
            validation.status
            in {
                TelemetryStatus.WARNING,
                TelemetryStatus.CRITICAL,
            }
        ):
            critical = (
                validation.status
                == TelemetryStatus.CRITICAL
            )

            evidence_type = (
                self._evidence_type(
                    point.sensor_type
                )
            )

            evidence.append(
                TelemetryEvidence(
                    evidence_id=new_id(),

                    evidence_type=(
                        evidence_type
                    ),

                    asset_id=point.asset_id,

                    description=(
                        f"Abnormal "
                        f"{point.sensor_type.value} "
                        f"reading: {point.value}"
                        f"{point.unit or ''}."
                    ),

                    severity=(
                        0.95
                        if critical
                        else 0.65
                    ),

                    confidence=0.95,

                    sensor_type=(
                        point.sensor_type
                    ),

                    sensor_id=point.sensor_id,
                    component=point.component,

                    value=point.value,
                    unit=point.unit,

                    timestamp=point.timestamp,

                    critical=critical,
                )
            )

        if (
            validation
            .rate_of_change_violation
        ):
            evidence.append(
                TelemetryEvidence(
                    evidence_id=new_id(),

                    evidence_type=(
                        TelemetryEvidenceType
                        .RAPID_CHANGE
                    ),

                    asset_id=point.asset_id,

                    description=(
                        f"{point.sensor_type.value} "
                        "changed faster than the "
                        "configured safe rate."
                    ),

                    severity=0.75,
                    confidence=0.90,

                    sensor_type=(
                        point.sensor_type
                    ),

                    sensor_id=point.sensor_id,
                    component=point.component,

                    value=point.value,
                    unit=point.unit,

                    timestamp=point.timestamp,
                )
            )

        if anomaly is not None:
            evidence.append(
                TelemetryEvidence(
                    evidence_id=new_id(),

                    evidence_type=(
                        TelemetryEvidenceType.ANOMALY
                    ),

                    asset_id=point.asset_id,

                    description=anomaly.reason,

                    severity=anomaly.score,

                    confidence=min(
                        0.95,
                        0.55
                        + anomaly.score * 0.40,
                    ),

                    sensor_type=(
                        point.sensor_type
                    ),

                    sensor_id=point.sensor_id,
                    component=point.component,

                    value=point.value,
                    unit=point.unit,

                    timestamp=point.timestamp,

                    critical=(
                        anomaly.level
                        == AnomalyLevel.CRITICAL
                    ),
                )
            )

        if (
            trend is not None
            and trend.direction
            in {
                TrendDirection
                .STRONGLY_INCREASING,

                TrendDirection
                .STRONGLY_DECREASING,
            }
            and trend.confidence >= 0.60
        ):
            evidence.append(
                TelemetryEvidence(
                    evidence_id=new_id(),

                    evidence_type=(
                        TelemetryEvidenceType.TREND
                    ),

                    asset_id=point.asset_id,

                    description=(
                        f"{point.sensor_type.value} "
                        f"shows a "
                        f"{trend.direction.value} "
                        "trend."
                    ),

                    severity=0.60,
                    confidence=trend.confidence,

                    sensor_type=(
                        point.sensor_type
                    ),

                    sensor_id=point.sensor_id,
                    component=point.component,

                    value=point.value,
                    unit=point.unit,

                    timestamp=point.timestamp,

                    metadata={
                        "slope_per_hour":
                            trend.slope_per_hour,

                        "normalized_slope":
                            trend.normalized_slope,
                    },
                )
            )

        return evidence

    # =====================================================
    # Full analysis
    # =====================================================

    def analyze(
        self,
        *,
        asset_id: str,
        sensor_type: SensorType,
        points: Sequence[TelemetryPoint],
        threshold: SensorThreshold | None = None,
        now: datetime | None = None,
    ) -> TelemetryAnalysisResult:

        relevant = [
            point
            for point in points
            if (
                point.asset_id == asset_id
                and point.sensor_type
                == sensor_type
            )
        ]

        relevant = sorted(
            relevant,
            key=lambda item: item.timestamp,
        )[
            -self.config
            .maximum_history_points:
        ]

        if not relevant:
            return TelemetryAnalysisResult(
                asset_id=asset_id,

                sensor_type=sensor_type,

                latest=None,

                validation=None,

                statistics=self.statistics([]),

                trend=self.trend([]),

                anomalies=[],

                evidence=[],

                status=TelemetryStatus.UNKNOWN,

                confidence=0.0,

                warnings=[
                    "No telemetry is available for "
                    "this sensor type."
                ],
            )

        latest = relevant[-1]

        previous = (
            relevant[-2]
            if len(relevant) >= 2
            else None
        )

        validation = self.validate_point(
            latest,
            threshold=threshold,
            previous=previous,
            now=now,
        )

        stats = self.statistics(relevant)

        trend = self.trend(relevant)

        anomalies = self.detect_anomalies(
            relevant,
            threshold=threshold,
        )

        latest_anomaly = None

        if anomalies:
            # Prefer an anomaly corresponding to the
            # most recent reading.
            matching = [
                item
                for item in anomalies
                if item.timestamp
                == latest.timestamp
            ]

            if matching:
                latest_anomaly = max(
                    matching,
                    key=lambda item: item.score,
                )

        evidence = self.build_evidence(
            point=latest,
            validation=validation,
            trend=trend,
            anomaly=latest_anomaly,
        )

        status = validation.status

        if (
            latest_anomaly is not None
            and latest_anomaly.level
            == AnomalyLevel.CRITICAL
        ):
            status = TelemetryStatus.CRITICAL

        elif (
            latest_anomaly is not None
            and latest_anomaly.level
            in {
                AnomalyLevel.HIGH,
                AnomalyLevel.MODERATE,
            }
            and status
            == TelemetryStatus.NORMAL
        ):
            status = TelemetryStatus.WARNING

        if not validation.valid:
            confidence = 0.20

        elif validation.stale:
            confidence = 0.40

        else:
            sample_factor = clamp(
                len(relevant) / 20.0
            )

            confidence = (
                0.65
                + 0.20 * sample_factor
                + 0.15 * trend.confidence
            )

        warnings: list[str] = []

        if validation.stale:
            warnings.append(
                "Latest telemetry is stale."
            )

        if (
            len(relevant)
            < self.config
            .minimum_statistics_samples
        ):
            warnings.append(
                "Limited telemetry history is "
                "available."
            )

        return TelemetryAnalysisResult(
            asset_id=asset_id,

            sensor_type=sensor_type,

            latest=latest,

            validation=validation,

            statistics=stats,

            trend=trend,

            anomalies=anomalies,

            evidence=evidence,

            status=status,

            confidence=round(
                clamp(confidence),
                4,
            ),

            warnings=warnings,
        )

    # =====================================================
    # Snapshot
    # =====================================================

    def build_snapshot(
        self,
        *,
        asset_id: str,
        points: Sequence[TelemetryPoint],
        thresholds: Mapping[
            SensorType,
            SensorThreshold,
        ] | None = None,
    ) -> AssetTelemetrySnapshot:

        thresholds = thresholds or {}

        grouped: dict[
            SensorType,
            list[TelemetryPoint],
        ] = {}

        for point in points:
            if point.asset_id != asset_id:
                continue

            grouped.setdefault(
                point.sensor_type,
                [],
            ).append(point)

        values: dict[
            SensorType,
            float,
        ] = {}

        statuses: list[
            TelemetryStatus
        ] = []

        anomaly_scores: list[float] = []
        quality_scores: list[float] = []

        latest_time: datetime | None = None

        for sensor_type, items in grouped.items():
            analysis = self.analyze(
                asset_id=asset_id,
                sensor_type=sensor_type,
                points=items,
                threshold=thresholds.get(
                    sensor_type
                ),
            )

            if analysis.latest is None:
                continue

            values[sensor_type] = (
                analysis.latest.value
            )

            statuses.append(
                analysis.status
            )

            quality_scores.append(
                analysis.confidence
            )

            if analysis.anomalies:
                anomaly_scores.append(
                    max(
                        item.score
                        for item
                        in analysis.anomalies
                    )
                )

            if (
                latest_time is None
                or analysis.latest.timestamp
                > latest_time
            ):
                latest_time = (
                    analysis.latest.timestamp
                )

        if (
            TelemetryStatus.CRITICAL
            in statuses
        ):
            overall_status = (
                TelemetryStatus.CRITICAL
            )

        elif (
            TelemetryStatus.WARNING
            in statuses
        ):
            overall_status = (
                TelemetryStatus.WARNING
            )

        elif statuses:
            overall_status = (
                TelemetryStatus.NORMAL
            )

        else:
            overall_status = (
                TelemetryStatus.UNKNOWN
            )

        return AssetTelemetrySnapshot(
            asset_id=asset_id,

            timestamp=(
                latest_time or utc_now()
            ),

            temperature=values.get(
                SensorType.TEMPERATURE
            ),

            vibration=values.get(
                SensorType.VIBRATION
            ),

            pressure=values.get(
                SensorType.PRESSURE
            ),

            current=values.get(
                SensorType.CURRENT
            ),

            voltage=values.get(
                SensorType.VOLTAGE
            ),

            power=values.get(
                SensorType.POWER
            ),

            energy=values.get(
                SensorType.ENERGY
            ),

            speed=values.get(
                SensorType.SPEED
            ),

            rpm=values.get(
                SensorType.RPM
            ),

            flow=values.get(
                SensorType.FLOW
            ),

            status=overall_status,

            quality_score=round(
                (
                    statistics.fmean(
                        quality_scores
                    )
                    if quality_scores
                    else 0.0
                ),
                4,
            ),

            anomaly_score=round(
                (
                    max(anomaly_scores)
                    if anomaly_scores
                    else 0.0
                ),
                4,
            ),
        )

    # =====================================================
    # Multivariate composite risk
    # =====================================================

    def composite_telemetry_risk(
        self,
        analyses: Sequence[
            TelemetryAnalysisResult
        ],
    ) -> dict[str, Any]:

        if not analyses:
            return {
                "risk_score": 0.0,
                "confidence": 0.0,
                "status":
                    TelemetryStatus.UNKNOWN.value,
                "critical_sensors": [],
                "warning_sensors": [],
            }

        weighted_sum = 0.0
        total_weight = 0.0

        critical: list[str] = []
        warning: list[str] = []

        for analysis in analyses:
            if analysis.status == (
                TelemetryStatus.CRITICAL
            ):
                score = 1.0
                critical.append(
                    analysis.sensor_type.value
                )

            elif analysis.status == (
                TelemetryStatus.WARNING
            ):
                score = 0.65
                warning.append(
                    analysis.sensor_type.value
                )

            elif analysis.status == (
                TelemetryStatus.NORMAL
            ):
                score = 0.10

            else:
                # Unknown data must not be treated
                # as healthy.
                score = 0.50

            if analysis.anomalies:
                score = max(
                    score,
                    max(
                        item.score
                        for item
                        in analysis.anomalies
                    ),
                )

            weight = max(
                0.20,
                analysis.confidence,
            )

            weighted_sum += score * weight
            total_weight += weight

        risk = (
            weighted_sum / total_weight
            if total_weight
            else 0.0
        )

        if critical:
            risk = max(risk, 0.90)

        if risk >= 0.90:
            status = TelemetryStatus.CRITICAL

        elif risk >= 0.60:
            status = TelemetryStatus.WARNING

        else:
            status = TelemetryStatus.NORMAL

        confidence = statistics.fmean(
            analysis.confidence
            for analysis in analyses
        )

        return {
            "risk_score": round(
                clamp(risk),
                4,
            ),

            "confidence": round(
                clamp(confidence),
                4,
            ),

            "status": status.value,

            "critical_sensors": critical,

            "warning_sensors": warning,
        }

    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _ordered_valid_points(
        points: Sequence[TelemetryPoint],
    ) -> list[TelemetryPoint]:

        return sorted(
            [
                point
                for point in points
                if (
                    point.quality
                    not in {
                        TelemetryQuality.INVALID,
                        TelemetryQuality.BAD,
                        TelemetryQuality.MISSING,
                    }
                    and math.isfinite(
                        finite(
                            point.value,
                            default=float("nan"),
                        )
                    )
                )
            ],
            key=lambda item: item.timestamp,
        )

    @staticmethod
    def _evidence_type(
        sensor_type: SensorType,
    ) -> TelemetryEvidenceType:

        mapping = {
            SensorType.TEMPERATURE:
                TelemetryEvidenceType
                .HIGH_TEMPERATURE,

            SensorType.VIBRATION:
                TelemetryEvidenceType
                .HIGH_VIBRATION,

            SensorType.PRESSURE:
                TelemetryEvidenceType
                .PRESSURE_ANOMALY,

            SensorType.CURRENT:
                TelemetryEvidenceType
                .CURRENT_ANOMALY,

            SensorType.VOLTAGE:
                TelemetryEvidenceType
                .VOLTAGE_ANOMALY,

            SensorType.ENERGY:
                TelemetryEvidenceType
                .ENERGY_ANOMALY,
        }

        return mapping.get(
            sensor_type,
            TelemetryEvidenceType.ANOMALY,
        )


# =========================================================
# Default instance
# =========================================================

telemetry_tools = TelemetryTools()


# =========================================================
# Agent-facing functions
# =========================================================

def validate_telemetry(
    point: TelemetryPoint,
    *,
    threshold: SensorThreshold | None = None,
    previous: TelemetryPoint | None = None,
) -> TelemetryValidationResult:

    return telemetry_tools.validate_point(
        point,
        threshold=threshold,
        previous=previous,
    )


def calculate_telemetry_statistics(
    points: Sequence[TelemetryPoint],
) -> TelemetryStatistics:

    return telemetry_tools.statistics(
        points
    )


def calculate_telemetry_trend(
    points: Sequence[TelemetryPoint],
) -> TelemetryTrend:

    return telemetry_tools.trend(
        points
    )


def detect_telemetry_anomalies(
    points: Sequence[TelemetryPoint],
    *,
    threshold: SensorThreshold | None = None,
) -> list[TelemetryAnomaly]:

    return telemetry_tools.detect_anomalies(
        points,
        threshold=threshold,
    )


def analyze_telemetry(
    *,
    asset_id: str,
    sensor_type: SensorType,
    points: Sequence[TelemetryPoint],
    threshold: SensorThreshold | None = None,
) -> TelemetryAnalysisResult:

    return telemetry_tools.analyze(
        asset_id=asset_id,
        sensor_type=sensor_type,
        points=points,
        threshold=threshold,
    )


def build_asset_telemetry_snapshot(
    *,
    asset_id: str,
    points: Sequence[TelemetryPoint],
    thresholds: Mapping[
        SensorType,
        SensorThreshold,
    ] | None = None,
) -> AssetTelemetrySnapshot:

    return telemetry_tools.build_snapshot(
        asset_id=asset_id,
        points=points,
        thresholds=thresholds,
    )


def calculate_telemetry_quality(
    *,
    asset_id: str,
    points: Sequence[TelemetryPoint],
    expected_points: int | None = None,
) -> TelemetryQualityReport:

    return telemetry_tools.quality_report(
        points,
        asset_id=asset_id,
        expected_points=expected_points,
    )


def calculate_composite_telemetry_risk(
    analyses: Sequence[
        TelemetryAnalysisResult
    ],
) -> dict[str, Any]:

    return (
        telemetry_tools
        .composite_telemetry_risk(
            analyses
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "SensorType",
    "TelemetryQuality",
    "TelemetryStatus",
    "TrendDirection",
    "AnomalyLevel",
    "TelemetryEvidenceType",

    "SensorThreshold",
    "TelemetryPoint",

    "TelemetryValidationResult",
    "TelemetryStatistics",
    "TelemetryTrend",
    "TelemetryAnomaly",
    "TelemetryEvidence",

    "TelemetryQualityReport",
    "AssetTelemetrySnapshot",
    "TelemetryAnalysisResult",

    "TelemetryToolsConfig",
    "TelemetryTools",

    "telemetry_tools",

    "validate_telemetry",
    "calculate_telemetry_statistics",
    "calculate_telemetry_trend",
    "detect_telemetry_anomalies",
    "analyze_telemetry",
    "build_asset_telemetry_snapshot",
    "calculate_telemetry_quality",
    "calculate_composite_telemetry_risk",
]