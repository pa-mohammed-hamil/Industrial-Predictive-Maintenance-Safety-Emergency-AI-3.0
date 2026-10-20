# app/observability/data_drift.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite, log, sqrt
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


# =========================================================
# Enums
# =========================================================

class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DriftStatus(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFTED = "drifted"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class FeatureType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class DriftMetric(str, Enum):
    PSI = "psi"
    KS = "ks"
    MEAN_SHIFT = "mean_shift"
    STD_SHIFT = "std_shift"
    MISSING_RATE = "missing_rate"
    CATEGORICAL_TVD = "categorical_tvd"


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class DriftThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.25

    ks_warning: float = 0.10
    ks_critical: float = 0.20

    mean_shift_warning: float = 0.50
    mean_shift_critical: float = 1.00

    std_shift_warning: float = 0.25
    std_shift_critical: float = 0.50

    missing_rate_warning: float = 0.05
    missing_rate_critical: float = 0.15

    categorical_tvd_warning: float = 0.10
    categorical_tvd_critical: float = 0.25


@dataclass(slots=True)
class DriftConfig:
    bins: int = 10

    minimum_reference_samples: int = 30
    minimum_current_samples: int = 30

    epsilon: float = 1e-6

    thresholds: DriftThresholds = field(
        default_factory=DriftThresholds
    )

    critical_feature_ratio: float = 0.20
    drifted_feature_ratio: float = 0.30


# =========================================================
# Feature configuration
# =========================================================

@dataclass(slots=True)
class FeatureSpec:
    name: str

    feature_type: FeatureType = (
        FeatureType.NUMERIC
    )

    critical: bool = False

    weight: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Metric structures
# =========================================================

@dataclass(slots=True)
class NumericDriftMetrics:
    psi: float | None = None
    ks_distance: float | None = None

    reference_mean: float | None = None
    current_mean: float | None = None

    reference_std: float | None = None
    current_std: float | None = None

    normalized_mean_shift: float | None = None
    normalized_std_shift: float | None = None

    reference_missing_rate: float = 0.0
    current_missing_rate: float = 0.0

    missing_rate_shift: float = 0.0


@dataclass(slots=True)
class CategoricalDriftMetrics:
    total_variation_distance: float | None = None

    reference_distribution: dict[
        str,
        float,
    ] = field(default_factory=dict)

    current_distribution: dict[
        str,
        float,
    ] = field(default_factory=dict)

    reference_missing_rate: float = 0.0
    current_missing_rate: float = 0.0

    missing_rate_shift: float = 0.0


# =========================================================
# Feature result
# =========================================================

@dataclass(slots=True)
class FeatureDriftResult:
    feature_name: str
    feature_type: FeatureType

    reference_samples: int
    current_samples: int

    reference_valid_samples: int
    current_valid_samples: int

    status: DriftStatus
    severity: DriftSeverity

    drift_detected: bool

    drift_score: float

    numeric_metrics: NumericDriftMetrics | None = None

    categorical_metrics: (
        CategoricalDriftMetrics | None
    ) = None

    triggered_metrics: list[str] = field(
        default_factory=list
    )

    critical_feature: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# Dataset result
# =========================================================

@dataclass(slots=True)
class DataDriftResult:
    evaluated_at: datetime

    total_features: int
    evaluated_features: int

    stable_features: int
    warning_features: int
    drifted_features: int
    critical_features: int

    drifted_feature_ratio: float

    weighted_drift_score: float

    status: DriftStatus
    severity: DriftSeverity

    drift_detected: bool

    feature_results: dict[
        str,
        FeatureDriftResult,
    ]

    errors: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["evaluated_at"] = (
            self.evaluated_at.isoformat()
        )

        result["status"] = self.status.value
        result["severity"] = self.severity.value

        result["feature_results"] = {
            name: feature.to_dict()
            for name, feature
            in self.feature_results.items()
        }

        return result


# =========================================================
# Helpers
# =========================================================

def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _finite_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not isfinite(result):
        return None

    return result


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# =========================================================
# Missing-rate drift
# =========================================================

def calculate_missing_rate(
    values: Sequence[Any],
) -> float:

    if not values:
        return 0.0

    missing = sum(
        value is None
        or (
            isinstance(
                value,
                float,
            )
            and not isfinite(value)
        )
        for value in values
    )

    return round(
        missing / len(values),
        6,
    )


# =========================================================
# Numeric preprocessing
# =========================================================

def clean_numeric_values(
    values: Iterable[Any],
) -> list[float]:

    cleaned: list[float] = []

    for value in values:
        parsed = _finite_float(
            value
        )

        if parsed is not None:
            cleaned.append(
                parsed
            )

    return cleaned


# =========================================================
# Quantile helper
# =========================================================

def _quantile(
    values: Sequence[float],
    q: float,
) -> float:

    if not values:
        raise ValueError(
            "Cannot calculate quantile "
            "of empty values."
        )

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    q = _clamp(q)

    position = (
        q
        * (
            len(ordered)
            - 1
        )
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


# =========================================================
# PSI
# =========================================================

def calculate_psi(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float | None:
    """
    Population Stability Index.

    Typical interpretation:

        < 0.10  stable
        0.10-0.25 moderate drift
        > 0.25 significant drift
    """

    if (
        not reference
        or not current
    ):
        return None

    if bins < 2:
        raise ValueError(
            "bins must be >= 2."
        )

    boundaries = [
        _quantile(
            reference,
            index / bins,
        )
        for index in range(
            1,
            bins,
        )
    ]

    # Remove duplicate boundaries caused by
    # low-cardinality features.
    boundaries = sorted(
        set(boundaries)
    )

    def bucket_counts(
        values: Sequence[float],
    ) -> list[int]:

        counts = [
            0
            for _ in range(
                len(boundaries) + 1
            )
        ]

        for value in values:
            index = 0

            while (
                index < len(boundaries)
                and value > boundaries[index]
            ):
                index += 1

            counts[index] += 1

        return counts

    reference_counts = bucket_counts(
        reference
    )

    current_counts = bucket_counts(
        current
    )

    reference_total = len(reference)
    current_total = len(current)

    psi = 0.0

    for ref_count, cur_count in zip(
        reference_counts,
        current_counts,
    ):
        ref_ratio = max(
            ref_count
            / reference_total,
            epsilon,
        )

        cur_ratio = max(
            cur_count
            / current_total,
            epsilon,
        )

        psi += (
            cur_ratio
            - ref_ratio
        ) * log(
            cur_ratio
            / ref_ratio
        )

    return round(
        psi,
        6,
    )


# =========================================================
# KS distance
# =========================================================

def calculate_ks_distance(
    reference: Sequence[float],
    current: Sequence[float],
) -> float | None:
    """
    Two-sample Kolmogorov-Smirnov distance.

    Returns D statistic only.
    No scipy dependency is required.
    """

    if (
        not reference
        or not current
    ):
        return None

    ref = sorted(reference)
    cur = sorted(current)

    values = sorted(
        set(
            ref + cur
        )
    )

    ref_index = 0
    cur_index = 0

    maximum = 0.0

    for value in values:

        while (
            ref_index < len(ref)
            and ref[ref_index] <= value
        ):
            ref_index += 1

        while (
            cur_index < len(cur)
            and cur[cur_index] <= value
        ):
            cur_index += 1

        ref_cdf = (
            ref_index / len(ref)
        )

        cur_cdf = (
            cur_index / len(cur)
        )

        maximum = max(
            maximum,
            abs(
                ref_cdf
                - cur_cdf
            ),
        )

    return round(
        maximum,
        6,
    )


# =========================================================
# Mean / std drift
# =========================================================

def calculate_normalized_mean_shift(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> float | None:

    if (
        not reference
        or not current
    ):
        return None

    reference_mean = mean(
        reference
    )

    current_mean = mean(
        current
    )

    reference_std = (
        pstdev(reference)
        if len(reference) > 1
        else 0.0
    )

    denominator = max(
        reference_std,
        epsilon,
    )

    return round(
        abs(
            current_mean
            - reference_mean
        )
        / denominator,
        6,
    )


def calculate_normalized_std_shift(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> float | None:

    if (
        len(reference) < 2
        or len(current) < 2
    ):
        return None

    reference_std = pstdev(
        reference
    )

    current_std = pstdev(
        current
    )

    return round(
        abs(
            current_std
            - reference_std
        )
        / max(
            reference_std,
            epsilon,
        ),
        6,
    )


# =========================================================
# Categorical distributions
# =========================================================

def calculate_category_distribution(
    values: Sequence[Any],
) -> dict[str, float]:

    valid = [
        str(value)
        for value in values
        if value is not None
    ]

    if not valid:
        return {}

    counts: dict[
        str,
        int,
    ] = {}

    for value in valid:
        counts[value] = (
            counts.get(
                value,
                0,
            )
            + 1
        )

    total = len(valid)

    return {
        category: round(
            count / total,
            6,
        )
        for category, count
        in counts.items()
    }


def calculate_total_variation_distance(
    reference: Mapping[str, float],
    current: Mapping[str, float],
) -> float:

    categories = (
        set(reference)
        | set(current)
    )

    distance = 0.5 * sum(
        abs(
            reference.get(
                category,
                0.0,
            )
            - current.get(
                category,
                0.0,
            )
        )
        for category in categories
    )

    return round(
        distance,
        6,
    )


# =========================================================
# Threshold scoring
# =========================================================

def _metric_severity(
    value: float | None,
    *,
    warning: float,
    critical: float,
) -> DriftSeverity:

    if value is None:
        return DriftSeverity.NONE

    if value >= critical:
        return DriftSeverity.CRITICAL

    if value >= warning:
        return DriftSeverity.MODERATE

    return DriftSeverity.NONE


def _severity_rank(
    severity: DriftSeverity,
) -> int:

    return {
        DriftSeverity.NONE: 0,
        DriftSeverity.LOW: 1,
        DriftSeverity.MODERATE: 2,
        DriftSeverity.HIGH: 3,
        DriftSeverity.CRITICAL: 4,
    }[severity]


def _max_severity(
    severities: Iterable[
        DriftSeverity
    ],
) -> DriftSeverity:

    return max(
        severities,
        key=_severity_rank,
        default=DriftSeverity.NONE,
    )


def _normalize_metric_score(
    value: float | None,
    *,
    warning: float,
    critical: float,
) -> float:

    if value is None:
        return 0.0

    if critical <= 0:
        return 0.0

    return _clamp(
        value / critical
    )


# =========================================================
# Numeric feature drift
# =========================================================

def evaluate_numeric_feature(
    feature: FeatureSpec,
    reference_values: Sequence[Any],
    current_values: Sequence[Any],
    *,
    config: DriftConfig,
) -> FeatureDriftResult:

    reference = clean_numeric_values(
        reference_values
    )

    current = clean_numeric_values(
        current_values
    )

    reference_missing = (
        calculate_missing_rate(
            reference_values
        )
    )

    current_missing = (
        calculate_missing_rate(
            current_values
        )
    )

    missing_shift = abs(
        current_missing
        - reference_missing
    )

    if (
        len(reference)
        < config.minimum_reference_samples
        or len(current)
        < config.minimum_current_samples
    ):
        return FeatureDriftResult(
            feature_name=feature.name,
            feature_type=feature.feature_type,

            reference_samples=len(
                reference_values
            ),

            current_samples=len(
                current_values
            ),

            reference_valid_samples=len(
                reference
            ),

            current_valid_samples=len(
                current
            ),

            status=(
                DriftStatus
                .INSUFFICIENT_DATA
            ),

            severity=DriftSeverity.NONE,

            drift_detected=False,
            drift_score=0.0,

            critical_feature=(
                feature.critical
            ),

            metadata=feature.metadata,
        )

    thresholds = config.thresholds

    psi = calculate_psi(
        reference,
        current,
        bins=config.bins,
        epsilon=config.epsilon,
    )

    ks = calculate_ks_distance(
        reference,
        current,
    )

    mean_shift = (
        calculate_normalized_mean_shift(
            reference,
            current,
            epsilon=config.epsilon,
        )
    )

    std_shift = (
        calculate_normalized_std_shift(
            reference,
            current,
            epsilon=config.epsilon,
        )
    )

    numeric_metrics = NumericDriftMetrics(
        psi=psi,
        ks_distance=ks,

        reference_mean=round(
            mean(reference),
            6,
        ),

        current_mean=round(
            mean(current),
            6,
        ),

        reference_std=round(
            pstdev(reference)
            if len(reference) > 1
            else 0.0,
            6,
        ),

        current_std=round(
            pstdev(current)
            if len(current) > 1
            else 0.0,
            6,
        ),

        normalized_mean_shift=(
            mean_shift
        ),

        normalized_std_shift=(
            std_shift
        ),

        reference_missing_rate=(
            reference_missing
        ),

        current_missing_rate=(
            current_missing
        ),

        missing_rate_shift=round(
            missing_shift,
            6,
        ),
    )

    metric_definitions = {
        DriftMetric.PSI.value: (
            psi,
            thresholds.psi_warning,
            thresholds.psi_critical,
        ),

        DriftMetric.KS.value: (
            ks,
            thresholds.ks_warning,
            thresholds.ks_critical,
        ),

        DriftMetric.MEAN_SHIFT.value: (
            mean_shift,
            thresholds.mean_shift_warning,
            thresholds.mean_shift_critical,
        ),

        DriftMetric.STD_SHIFT.value: (
            std_shift,
            thresholds.std_shift_warning,
            thresholds.std_shift_critical,
        ),

        DriftMetric.MISSING_RATE.value: (
            missing_shift,
            thresholds.missing_rate_warning,
            thresholds.missing_rate_critical,
        ),
    }

    severities: list[
        DriftSeverity
    ] = []

    triggered: list[str] = []

    scores: list[float] = []

    for name, (
        value,
        warning,
        critical,
    ) in metric_definitions.items():

        severity = _metric_severity(
            value,
            warning=warning,
            critical=critical,
        )

        severities.append(
            severity
        )

        if severity != DriftSeverity.NONE:
            triggered.append(
                name
            )

        scores.append(
            _normalize_metric_score(
                value,
                warning=warning,
                critical=critical,
            )
        )

    severity = _max_severity(
        severities
    )

    drift_detected = (
        severity
        in {
            DriftSeverity.MODERATE,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        }
    )

    if severity == DriftSeverity.CRITICAL:
        status = DriftStatus.CRITICAL

    elif drift_detected:
        status = DriftStatus.DRIFTED

    elif triggered:
        status = DriftStatus.WARNING

    else:
        status = DriftStatus.STABLE

    return FeatureDriftResult(
        feature_name=feature.name,
        feature_type=feature.feature_type,

        reference_samples=len(
            reference_values
        ),

        current_samples=len(
            current_values
        ),

        reference_valid_samples=len(
            reference
        ),

        current_valid_samples=len(
            current
        ),

        status=status,
        severity=severity,

        drift_detected=(
            drift_detected
        ),

        drift_score=round(
            max(scores)
            if scores
            else 0.0,
            6,
        ),

        numeric_metrics=(
            numeric_metrics
        ),

        triggered_metrics=(
            triggered
        ),

        critical_feature=(
            feature.critical
        ),

        metadata=feature.metadata,
    )


# =========================================================
# Categorical feature drift
# =========================================================

def evaluate_categorical_feature(
    feature: FeatureSpec,
    reference_values: Sequence[Any],
    current_values: Sequence[Any],
    *,
    config: DriftConfig,
) -> FeatureDriftResult:

    reference_valid = [
        value
        for value in reference_values
        if value is not None
    ]

    current_valid = [
        value
        for value in current_values
        if value is not None
    ]

    reference_missing = (
        calculate_missing_rate(
            reference_values
        )
    )

    current_missing = (
        calculate_missing_rate(
            current_values
        )
    )

    missing_shift = abs(
        current_missing
        - reference_missing
    )

    if (
        len(reference_valid)
        < config.minimum_reference_samples
        or len(current_valid)
        < config.minimum_current_samples
    ):
        return FeatureDriftResult(
            feature_name=feature.name,
            feature_type=feature.feature_type,

            reference_samples=len(
                reference_values
            ),

            current_samples=len(
                current_values
            ),

            reference_valid_samples=len(
                reference_valid
            ),

            current_valid_samples=len(
                current_valid
            ),

            status=(
                DriftStatus
                .INSUFFICIENT_DATA
            ),

            severity=DriftSeverity.NONE,

            drift_detected=False,
            drift_score=0.0,

            critical_feature=(
                feature.critical
            ),

            metadata=feature.metadata,
        )

    reference_distribution = (
        calculate_category_distribution(
            reference_values
        )
    )

    current_distribution = (
        calculate_category_distribution(
            current_values
        )
    )

    tvd = (
        calculate_total_variation_distance(
            reference_distribution,
            current_distribution,
        )
    )

    thresholds = config.thresholds

    tvd_severity = _metric_severity(
        tvd,
        warning=(
            thresholds
            .categorical_tvd_warning
        ),
        critical=(
            thresholds
            .categorical_tvd_critical
        ),
    )

    missing_severity = _metric_severity(
        missing_shift,
        warning=(
            thresholds
            .missing_rate_warning
        ),
        critical=(
            thresholds
            .missing_rate_critical
        ),
    )

    severity = _max_severity(
        [
            tvd_severity,
            missing_severity,
        ]
    )

    triggered: list[str] = []

    if tvd_severity != DriftSeverity.NONE:
        triggered.append(
            DriftMetric
            .CATEGORICAL_TVD.value
        )

    if (
        missing_severity
        != DriftSeverity.NONE
    ):
        triggered.append(
            DriftMetric
            .MISSING_RATE.value
        )

    drift_detected = (
        severity
        in {
            DriftSeverity.MODERATE,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        }
    )

    if severity == DriftSeverity.CRITICAL:
        status = DriftStatus.CRITICAL

    elif drift_detected:
        status = DriftStatus.DRIFTED

    elif triggered:
        status = DriftStatus.WARNING

    else:
        status = DriftStatus.STABLE

    drift_score = max(
        _normalize_metric_score(
            tvd,
            warning=(
                thresholds
                .categorical_tvd_warning
            ),
            critical=(
                thresholds
                .categorical_tvd_critical
            ),
        ),
        _normalize_metric_score(
            missing_shift,
            warning=(
                thresholds
                .missing_rate_warning
            ),
            critical=(
                thresholds
                .missing_rate_critical
            ),
        ),
    )

    return FeatureDriftResult(
        feature_name=feature.name,
        feature_type=feature.feature_type,

        reference_samples=len(
            reference_values
        ),

        current_samples=len(
            current_values
        ),

        reference_valid_samples=len(
            reference_valid
        ),

        current_valid_samples=len(
            current_valid
        ),

        status=status,
        severity=severity,

        drift_detected=(
            drift_detected
        ),

        drift_score=round(
            drift_score,
            6,
        ),

        categorical_metrics=(
            CategoricalDriftMetrics(
                total_variation_distance=tvd,

                reference_distribution=(
                    reference_distribution
                ),

                current_distribution=(
                    current_distribution
                ),

                reference_missing_rate=(
                    reference_missing
                ),

                current_missing_rate=(
                    current_missing
                ),

                missing_rate_shift=round(
                    missing_shift,
                    6,
                ),
            )
        ),

        triggered_metrics=(
            triggered
        ),

        critical_feature=(
            feature.critical
        ),

        metadata=feature.metadata,
    )


# =========================================================
# Dataset-level drift
# =========================================================

def evaluate_data_drift(
    *,
    reference_data: Mapping[
        str,
        Sequence[Any],
    ],
    current_data: Mapping[
        str,
        Sequence[Any],
    ],
    features: Sequence[FeatureSpec],
    config: DriftConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> DataDriftResult:

    config = config or DriftConfig()

    feature_results: dict[
        str,
        FeatureDriftResult,
    ] = {}

    errors: list[
        dict[str, Any]
    ] = []

    for feature in features:

        reference_values = (
            reference_data.get(
                feature.name
            )
        )

        current_values = (
            current_data.get(
                feature.name
            )
        )

        if reference_values is None:
            errors.append(
                {
                    "feature":
                        feature.name,

                    "error":
                        "reference_feature_missing",
                }
            )
            continue

        if current_values is None:
            errors.append(
                {
                    "feature":
                        feature.name,

                    "error":
                        "current_feature_missing",
                }
            )
            continue

        try:
            if (
                feature.feature_type
                == FeatureType.NUMERIC
            ):
                result = (
                    evaluate_numeric_feature(
                        feature,
                        reference_values,
                        current_values,
                        config=config,
                    )
                )

            else:
                result = (
                    evaluate_categorical_feature(
                        feature,
                        reference_values,
                        current_values,
                        config=config,
                    )
                )

            feature_results[
                feature.name
            ] = result

        except Exception as exc:
            errors.append(
                {
                    "feature":
                        feature.name,

                    "error":
                        type(exc).__name__,

                    "message":
                        str(exc),
                }
            )

    evaluated = [
        result
        for result
        in feature_results.values()
        if (
            result.status
            != DriftStatus
            .INSUFFICIENT_DATA
        )
    ]

    stable = sum(
        result.status
        == DriftStatus.STABLE
        for result in evaluated
    )

    warning = sum(
        result.status
        == DriftStatus.WARNING
        for result in evaluated
    )

    drifted = sum(
        result.status
        == DriftStatus.DRIFTED
        for result in evaluated
    )

    critical = sum(
        result.status
        == DriftStatus.CRITICAL
        for result in evaluated
    )

    drift_count = (
        drifted + critical
    )

    drift_ratio = (
        drift_count / len(evaluated)
        if evaluated
        else 0.0
    )

    # ---------------------------------------------
    # Weighted drift
    # ---------------------------------------------

    weighted_sum = 0.0
    total_weight = 0.0

    spec_map = {
        feature.name: feature
        for feature in features
    }

    for result in evaluated:
        spec = spec_map[
            result.feature_name
        ]

        weight = max(
            0.0,
            spec.weight,
        )

        if spec.critical:
            weight *= 2.0

        weighted_sum += (
            result.drift_score
            * weight
        )

        total_weight += weight

    weighted_score = (
        weighted_sum / total_weight
        if total_weight
        else 0.0
    )

    # ---------------------------------------------
    # Dataset severity
    # ---------------------------------------------

    critical_feature_drift = any(
        result.critical_feature
        and result.drift_detected
        for result in evaluated
    )

    if critical_feature_drift:
        severity = DriftSeverity.CRITICAL
        status = DriftStatus.CRITICAL

    elif (
        drift_ratio
        >= config.drifted_feature_ratio
    ):
        severity = DriftSeverity.HIGH
        status = DriftStatus.DRIFTED

    elif drift_count > 0:
        severity = DriftSeverity.MODERATE
        status = DriftStatus.DRIFTED

    elif warning > 0:
        severity = DriftSeverity.LOW
        status = DriftStatus.WARNING

    else:
        severity = DriftSeverity.NONE
        status = DriftStatus.STABLE

    return DataDriftResult(
        evaluated_at=utc_now(),

        total_features=len(
            features
        ),

        evaluated_features=len(
            evaluated
        ),

        stable_features=stable,
        warning_features=warning,
        drifted_features=drifted,
        critical_features=critical,

        drifted_feature_ratio=round(
            drift_ratio,
            6,
        ),

        weighted_drift_score=round(
            weighted_score,
            6,
        ),

        status=status,
        severity=severity,

        drift_detected=(
            drift_count > 0
            or critical_feature_drift
        ),

        feature_results=(
            feature_results
        ),

        errors=errors,

        metadata=(
            metadata or {}
        ),
    )


# =========================================================
# Quality gate
# =========================================================

def data_drift_quality_gate(
    result: DataDriftResult,
    *,
    maximum_drifted_feature_ratio: float = 0.20,
    maximum_weighted_drift_score: float = 0.50,
    allow_critical_feature_drift: bool = False,
) -> dict[str, Any]:

    critical_drift = any(
        feature.critical_feature
        and feature.drift_detected
        for feature
        in result.feature_results.values()
    )

    checks = {
        "drifted_feature_ratio": (
            result.drifted_feature_ratio
            <= maximum_drifted_feature_ratio
        ),

        "weighted_drift_score": (
            result.weighted_drift_score
            <= maximum_weighted_drift_score
        ),

        "critical_feature_drift": (
            allow_critical_feature_drift
            or not critical_drift
        ),
    }

    return {
        "passed": all(
            checks.values()
        ),

        "checks": checks,

        "metrics": {
            "drifted_feature_ratio":
                result.drifted_feature_ratio,

            "weighted_drift_score":
                result.weighted_drift_score,

            "drifted_features":
                result.drifted_features,

            "critical_features":
                result.critical_features,

            "status":
                result.status.value,

            "severity":
                result.severity.value,
        },
    }


# =========================================================
# Alert payload
# =========================================================

def build_drift_alert_payload(
    result: DataDriftResult,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:

    affected = [
        {
            "feature":
                feature.feature_name,

            "severity":
                feature.severity.value,

            "score":
                feature.drift_score,

            "critical":
                feature.critical_feature,

            "metrics":
                feature.triggered_metrics,
        }
        for feature
        in result.feature_results.values()
        if feature.drift_detected
    ]

    return {
        "event_type":
            "data_drift_detected",

        "severity":
            result.severity.value,

        "status":
            result.status.value,

        "model_name":
            model_name,

        "model_version":
            model_version,

        "drifted_feature_ratio":
            result.drifted_feature_ratio,

        "weighted_drift_score":
            result.weighted_drift_score,

        "affected_features":
            affected,

        "evaluated_at":
            result.evaluated_at.isoformat(),
    }


# =========================================================
# Monitor service
# =========================================================

class DataDriftMonitor:

    def __init__(
        self,
        *,
        config: DriftConfig | None = None,
    ) -> None:

        self.config = (
            config or DriftConfig()
        )

        self._reference_data: dict[
            str,
            list[Any],
        ] = {}

        self._features: list[
            FeatureSpec
        ] = []

        self._last_result: (
            DataDriftResult | None
        ) = None

    def set_reference(
        self,
        data: Mapping[
            str,
            Sequence[Any],
        ],
        *,
        features: Sequence[
            FeatureSpec
        ],
    ) -> None:

        self._reference_data = {
            name: list(values)
            for name, values
            in data.items()
        }

        self._features = list(
            features
        )

    def evaluate(
        self,
        current_data: Mapping[
            str,
            Sequence[Any],
        ],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DataDriftResult:

        if not self._reference_data:
            raise RuntimeError(
                "Reference data has not "
                "been configured."
            )

        if not self._features:
            raise RuntimeError(
                "Feature specifications have "
                "not been configured."
            )

        result = evaluate_data_drift(
            reference_data=(
                self._reference_data
            ),

            current_data=current_data,

            features=self._features,

            config=self.config,

            metadata=metadata,
        )

        self._last_result = result

        return result

    def quality_gate(
        self,
        result: DataDriftResult | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:

        target = (
            result
            or self._last_result
        )

        if target is None:
            raise RuntimeError(
                "No drift evaluation "
                "has been performed."
            )

        return data_drift_quality_gate(
            target,
            **kwargs,
        )

    @property
    def last_result(
        self,
    ) -> DataDriftResult | None:

        return self._last_result


data_drift_monitor = DataDriftMonitor()


__all__ = [
    "DriftSeverity",
    "DriftStatus",
    "FeatureType",
    "DriftMetric",
    "DriftThresholds",
    "DriftConfig",
    "FeatureSpec",
    "NumericDriftMetrics",
    "CategoricalDriftMetrics",
    "FeatureDriftResult",
    "DataDriftResult",
    "calculate_missing_rate",
    "clean_numeric_values",
    "calculate_psi",
    "calculate_ks_distance",
    "calculate_normalized_mean_shift",
    "calculate_normalized_std_shift",
    "calculate_category_distribution",
    "calculate_total_variation_distance",
    "evaluate_numeric_feature",
    "evaluate_categorical_feature",
    "evaluate_data_drift",
    "data_drift_quality_gate",
    "build_drift_alert_payload",
    "DataDriftMonitor",
    "data_drift_monitor",
]