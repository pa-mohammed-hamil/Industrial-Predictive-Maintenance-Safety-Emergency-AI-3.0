# app/observability/model_drift.py

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Iterable, Sequence


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _finite(
    value: float,
    *,
    name: str,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


def _probability(
    value: float,
    *,
    name: str,
) -> float:
    value = _finite(
        value,
        name=name,
    )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def _safe_mean(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return mean(values)


def _relative_change(
    baseline: float,
    current: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    denominator = max(
        abs(baseline),
        epsilon,
    )

    return (
        current - baseline
    ) / denominator


# =========================================================
# Enums
# =========================================================

class ModelTaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"

    ANOMALY_DETECTION = "anomaly_detection"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_PREDICTION = "rul_prediction"

    DEFECT_DETECTION = "defect_detection"
    FIRE_DETECTION = "fire_detection"
    SMOKE_DETECTION = "smoke_detection"

    OTHER = "other"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftStatus(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DRIFTED = "drifted"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class DriftMetric(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"

    FALSE_NEGATIVE_RATE = "false_negative_rate"
    FALSE_POSITIVE_RATE = "false_positive_rate"

    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"

    CALIBRATION_ERROR = "calibration_error"

    CONFIDENCE = "confidence"

    LATENCY = "latency"

    CUSTOM = "custom"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class DriftThreshold:
    warning: float = 0.10
    high: float = 0.20
    critical: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "warning",
            "high",
            "critical",
        ):
            value = getattr(
                self,
                name,
            )

            if value < 0:
                raise ValueError(
                    f"{name} must be >= 0."
                )

        if not (
            self.warning
            <= self.high
            <= self.critical
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "warning <= high <= critical."
            )


@dataclass(slots=True)
class ModelDriftConfig:
    minimum_samples: int = 30

    default_threshold: DriftThreshold = field(
        default_factory=DriftThreshold
    )

    accuracy_threshold: DriftThreshold = field(
        default_factory=lambda: DriftThreshold(
            warning=0.05,
            high=0.10,
            critical=0.20,
        )
    )

    recall_threshold: DriftThreshold = field(
        default_factory=lambda: DriftThreshold(
            warning=0.05,
            high=0.10,
            critical=0.20,
        )
    )

    false_negative_threshold: DriftThreshold = field(
        default_factory=lambda: DriftThreshold(
            warning=0.10,
            high=0.20,
            critical=0.35,
        )
    )

    error_threshold: DriftThreshold = field(
        default_factory=lambda: DriftThreshold(
            warning=0.10,
            high=0.20,
            critical=0.35,
        )
    )

    confidence_threshold: DriftThreshold = field(
        default_factory=lambda: DriftThreshold(
            warning=0.10,
            high=0.20,
            critical=0.30,
        )
    )

    latency_threshold: DriftThreshold = field(
        default_factory=lambda: DriftThreshold(
            warning=0.25,
            high=0.50,
            critical=1.00,
        )
    )

    critical_metric_weight: float = 2.0

    safety_critical_metrics: set[
        DriftMetric
    ] = field(
        default_factory=lambda: {
            DriftMetric.RECALL,
            DriftMetric.FALSE_NEGATIVE_RATE,
        }
    )


# =========================================================
# Observation
# =========================================================

@dataclass(slots=True)
class ModelObservation:
    model_name: str
    model_version: str

    task_type: ModelTaskType

    prediction: float | int | str
    actual: float | int | str | None = None

    confidence: float | None = None
    latency_ms: float | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    asset_id: str | None = None
    site_id: str | None = None

    prediction_id: str | None = None

    is_critical_case: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Metric snapshot
# =========================================================

@dataclass(slots=True)
class ModelMetricSnapshot:
    metric: DriftMetric

    value: float

    sample_count: int

    direction: MetricDirection

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Drift result
# =========================================================

@dataclass(slots=True)
class MetricDriftResult:
    metric: DriftMetric

    baseline_value: float
    current_value: float

    direction: MetricDirection

    relative_change: float
    degradation: float

    drift_score: float

    severity: DriftSeverity

    drift_detected: bool

    safety_critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ModelDriftResult:
    model_name: str
    model_version: str

    task_type: ModelTaskType

    baseline_samples: int
    current_samples: int

    status: DriftStatus
    severity: DriftSeverity

    drift_detected: bool

    overall_drift_score: float

    drifted_metrics: list[str]
    critical_metrics: list[str]

    metric_results: list[
        MetricDriftResult
    ]

    baseline_metrics: dict[str, float]
    current_metrics: dict[str, float]

    recommendation: str

    evaluated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["task_type"] = (
            self.task_type.value
        )

        data["status"] = (
            self.status.value
        )

        data["severity"] = (
            self.severity.value
        )

        data["evaluated_at"] = (
            self.evaluated_at.isoformat()
        )

        for result in data[
            "metric_results"
        ]:
            metric = result.get(
                "metric"
            )

            direction = result.get(
                "direction"
            )

            severity = result.get(
                "severity"
            )

            if isinstance(metric, Enum):
                result[
                    "metric"
                ] = metric.value

            if isinstance(
                direction,
                Enum,
            ):
                result[
                    "direction"
                ] = direction.value

            if isinstance(
                severity,
                Enum,
            ):
                result[
                    "severity"
                ] = severity.value

        return data


# =========================================================
# Classification metrics
# =========================================================

@dataclass(slots=True)
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    false_positive_rate: float
    false_negative_rate: float

    average_confidence: float | None

    samples: int

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


def calculate_binary_metrics(
    observations: Sequence[
        ModelObservation
    ],
    *,
    positive_label: Any = 1,
) -> BinaryClassificationMetrics:

    valid = [
        observation
        for observation in observations
        if observation.actual is not None
    ]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    confidences: list[float] = []

    for observation in valid:
        predicted_positive = (
            observation.prediction
            == positive_label
        )

        actual_positive = (
            observation.actual
            == positive_label
        )

        if (
            predicted_positive
            and actual_positive
        ):
            tp += 1

        elif (
            not predicted_positive
            and not actual_positive
        ):
            tn += 1

        elif (
            predicted_positive
            and not actual_positive
        ):
            fp += 1

        else:
            fn += 1

        if (
            observation.confidence
            is not None
        ):
            confidences.append(
                _probability(
                    observation.confidence,
                    name="confidence",
                )
            )

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if fn + tp
        else 0.0
    )

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,

        false_positive_rate=fpr,
        false_negative_rate=fnr,

        average_confidence=(
            _safe_mean(
                confidences
            )
        ),

        samples=total,

        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


# =========================================================
# Regression metrics
# =========================================================

@dataclass(slots=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float | None

    bias: float

    samples: int


def calculate_regression_metrics(
    observations: Sequence[
        ModelObservation
    ],
) -> RegressionMetrics:

    actual: list[float] = []
    predicted: list[float] = []

    for observation in observations:
        if observation.actual is None:
            continue

        try:
            actual_value = float(
                observation.actual
            )

            predicted_value = float(
                observation.prediction
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if not (
            math.isfinite(actual_value)
            and math.isfinite(
                predicted_value
            )
        ):
            continue

        actual.append(
            actual_value
        )

        predicted.append(
            predicted_value
        )

    if not actual:
        return RegressionMetrics(
            mae=0.0,
            rmse=0.0,
            mape=None,
            bias=0.0,
            samples=0,
        )

    errors = [
        predicted_value
        - actual_value
        for predicted_value, actual_value
        in zip(
            predicted,
            actual,
        )
    ]

    absolute_errors = [
        abs(error)
        for error in errors
    ]

    squared_errors = [
        error * error
        for error in errors
    ]

    percentage_errors = [
        abs(error)
        / abs(actual_value)
        for error, actual_value
        in zip(
            errors,
            actual,
        )
        if abs(actual_value) > 1e-12
    ]

    return RegressionMetrics(
        mae=mean(
            absolute_errors
        ),

        rmse=math.sqrt(
            mean(
                squared_errors
            )
        ),

        mape=(
            mean(
                percentage_errors
            )
            if percentage_errors
            else None
        ),

        bias=mean(errors),

        samples=len(actual),
    )


# =========================================================
# Generic accuracy
# =========================================================

def calculate_multiclass_accuracy(
    observations: Sequence[
        ModelObservation
    ],
) -> float:

    valid = [
        observation
        for observation in observations
        if observation.actual is not None
    ]

    if not valid:
        return 0.0

    correct = sum(
        observation.prediction
        == observation.actual
        for observation in valid
    )

    return correct / len(valid)


# =========================================================
# Latency
# =========================================================

def calculate_average_latency(
    observations: Sequence[
        ModelObservation
    ],
) -> float | None:

    values = [
        observation.latency_ms
        for observation in observations
        if (
            observation.latency_ms
            is not None
            and observation.latency_ms >= 0
        )
    ]

    return _safe_mean(
        values
    )


# =========================================================
# Snapshot creation
# =========================================================

def build_metric_snapshots(
    observations: Sequence[
        ModelObservation
    ],
    *,
    positive_label: Any = 1,
) -> dict[
    DriftMetric,
    ModelMetricSnapshot,
]:

    if not observations:
        return {}

    task = observations[
        0
    ].task_type

    snapshots: dict[
        DriftMetric,
        ModelMetricSnapshot,
    ] = {}

    # -----------------------------------------------------
    # Classification / failure / safety models
    # -----------------------------------------------------

    classification_tasks = {
        ModelTaskType.CLASSIFICATION,
        ModelTaskType.FAILURE_PREDICTION,
        ModelTaskType.ANOMALY_DETECTION,
        ModelTaskType.DEFECT_DETECTION,
        ModelTaskType.FIRE_DETECTION,
        ModelTaskType.SMOKE_DETECTION,
    }

    if task in classification_tasks:

        metrics = (
            calculate_binary_metrics(
                observations,
                positive_label=positive_label,
            )
        )

        values = {
            DriftMetric.ACCURACY:
                metrics.accuracy,

            DriftMetric.PRECISION:
                metrics.precision,

            DriftMetric.RECALL:
                metrics.recall,

            DriftMetric.F1:
                metrics.f1,

            DriftMetric.FALSE_POSITIVE_RATE:
                metrics.false_positive_rate,

            DriftMetric.FALSE_NEGATIVE_RATE:
                metrics.false_negative_rate,
        }

        for metric, value in values.items():

            direction = (
                MetricDirection
                .LOWER_IS_BETTER
                if metric
                in {
                    DriftMetric
                    .FALSE_POSITIVE_RATE,

                    DriftMetric
                    .FALSE_NEGATIVE_RATE,
                }
                else MetricDirection
                .HIGHER_IS_BETTER
            )

            snapshots[
                metric
            ] = ModelMetricSnapshot(
                metric=metric,
                value=value,
                sample_count=(
                    metrics.samples
                ),
                direction=direction,
            )

        if (
            metrics.average_confidence
            is not None
        ):
            snapshots[
                DriftMetric.CONFIDENCE
            ] = ModelMetricSnapshot(
                metric=(
                    DriftMetric.CONFIDENCE
                ),

                value=(
                    metrics
                    .average_confidence
                ),

                sample_count=(
                    metrics.samples
                ),

                direction=(
                    MetricDirection
                    .HIGHER_IS_BETTER
                ),
            )

    # -----------------------------------------------------
    # Regression / forecasting / RUL
    # -----------------------------------------------------

    regression_tasks = {
        ModelTaskType.REGRESSION,
        ModelTaskType.FORECASTING,
        ModelTaskType.RUL_PREDICTION,
    }

    if task in regression_tasks:

        metrics = (
            calculate_regression_metrics(
                observations
            )
        )

        snapshots[
            DriftMetric.MAE
        ] = ModelMetricSnapshot(
            metric=DriftMetric.MAE,
            value=metrics.mae,
            sample_count=metrics.samples,
            direction=(
                MetricDirection
                .LOWER_IS_BETTER
            ),
        )

        snapshots[
            DriftMetric.RMSE
        ] = ModelMetricSnapshot(
            metric=DriftMetric.RMSE,
            value=metrics.rmse,
            sample_count=metrics.samples,
            direction=(
                MetricDirection
                .LOWER_IS_BETTER
            ),
        )

        if metrics.mape is not None:
            snapshots[
                DriftMetric.MAPE
            ] = ModelMetricSnapshot(
                metric=DriftMetric.MAPE,

                value=metrics.mape,

                sample_count=(
                    metrics.samples
                ),

                direction=(
                    MetricDirection
                    .LOWER_IS_BETTER
                ),
            )

    # -----------------------------------------------------
    # Latency
    # -----------------------------------------------------

    latency = (
        calculate_average_latency(
            observations
        )
    )

    if latency is not None:
        snapshots[
            DriftMetric.LATENCY
        ] = ModelMetricSnapshot(
            metric=DriftMetric.LATENCY,
            value=latency,
            sample_count=len(
                observations
            ),
            direction=(
                MetricDirection
                .LOWER_IS_BETTER
            ),
        )

    return snapshots


# =========================================================
# Threshold selection
# =========================================================

def get_threshold(
    metric: DriftMetric,
    config: ModelDriftConfig,
) -> DriftThreshold:

    if metric == DriftMetric.ACCURACY:
        return config.accuracy_threshold

    if metric == DriftMetric.RECALL:
        return config.recall_threshold

    if (
        metric
        == DriftMetric.FALSE_NEGATIVE_RATE
    ):
        return (
            config.false_negative_threshold
        )

    if metric in {
        DriftMetric.MAE,
        DriftMetric.RMSE,
        DriftMetric.MAPE,
    }:
        return config.error_threshold

    if metric == DriftMetric.CONFIDENCE:
        return config.confidence_threshold

    if metric == DriftMetric.LATENCY:
        return config.latency_threshold

    return config.default_threshold


# =========================================================
# Degradation
# =========================================================

def calculate_degradation(
    *,
    baseline: float,
    current: float,
    direction: MetricDirection,
) -> float:

    relative = _relative_change(
        baseline,
        current,
    )

    if (
        direction
        == MetricDirection.HIGHER_IS_BETTER
    ):
        return max(
            0.0,
            -relative,
        )

    return max(
        0.0,
        relative,
    )


def classify_drift_severity(
    degradation: float,
    threshold: DriftThreshold,
) -> DriftSeverity:

    if degradation >= threshold.critical:
        return DriftSeverity.CRITICAL

    if degradation >= threshold.high:
        return DriftSeverity.HIGH

    if degradation >= threshold.warning:
        return DriftSeverity.MEDIUM

    if degradation > 0:
        return DriftSeverity.LOW

    return DriftSeverity.NONE


# =========================================================
# Metric comparison
# =========================================================

def compare_metric(
    *,
    baseline: ModelMetricSnapshot,
    current: ModelMetricSnapshot,
    config: ModelDriftConfig,
) -> MetricDriftResult:

    if baseline.metric != current.metric:
        raise ValueError(
            "Cannot compare different metrics."
        )

    metric = baseline.metric

    degradation = (
        calculate_degradation(
            baseline=baseline.value,
            current=current.value,
            direction=baseline.direction,
        )
    )

    relative_change = (
        _relative_change(
            baseline.value,
            current.value,
        )
    )

    threshold = get_threshold(
        metric,
        config,
    )

    severity = (
        classify_drift_severity(
            degradation,
            threshold,
        )
    )

    drift_detected = (
        degradation
        >= threshold.warning
    )

    safety_critical = (
        metric
        in config.safety_critical_metrics
    )

    drift_score = degradation

    if safety_critical:
        drift_score *= (
            config
            .critical_metric_weight
        )

    return MetricDriftResult(
        metric=metric,

        baseline_value=(
            baseline.value
        ),

        current_value=(
            current.value
        ),

        direction=(
            baseline.direction
        ),

        relative_change=(
            relative_change
        ),

        degradation=degradation,

        drift_score=drift_score,

        severity=severity,

        drift_detected=(
            drift_detected
        ),

        safety_critical=(
            safety_critical
        ),
    )


# =========================================================
# Overall severity
# =========================================================

_SEVERITY_RANK = {
    DriftSeverity.NONE: 0,
    DriftSeverity.LOW: 1,
    DriftSeverity.MEDIUM: 2,
    DriftSeverity.HIGH: 3,
    DriftSeverity.CRITICAL: 4,
}


def highest_severity(
    results: Sequence[
        MetricDriftResult
    ],
) -> DriftSeverity:

    if not results:
        return DriftSeverity.NONE

    return max(
        (
            result.severity
            for result in results
        ),
        key=lambda severity:
            _SEVERITY_RANK[
                severity
            ],
    )


def determine_status(
    severity: DriftSeverity,
) -> DriftStatus:

    if severity == DriftSeverity.CRITICAL:
        return DriftStatus.CRITICAL

    if severity in {
        DriftSeverity.HIGH,
        DriftSeverity.MEDIUM,
    }:
        return DriftStatus.DRIFTED

    if severity == DriftSeverity.LOW:
        return DriftStatus.WATCH

    return DriftStatus.HEALTHY


# =========================================================
# Recommendation
# =========================================================

def build_recommendation(
    result: Sequence[
        MetricDriftResult
    ],
) -> str:

    if not result:
        return (
            "No comparable model metrics "
            "were available."
        )

    critical = [
        metric
        for metric in result
        if (
            metric.severity
            == DriftSeverity.CRITICAL
        )
    ]

    high = [
        metric
        for metric in result
        if (
            metric.severity
            == DriftSeverity.HIGH
        )
    ]

    safety = [
        metric
        for metric in result
        if (
            metric.safety_critical
            and metric.drift_detected
        )
    ]

    if critical:
        return (
            "Critical model performance drift "
            "detected. Consider disabling or "
            "routing away from the affected model, "
            "validate recent predictions, investigate "
            "input/data drift, and retrain or roll back "
            "before normal production use."
        )

    if safety:
        return (
            "Safety-critical model degradation "
            "detected. Review false negatives and "
            "recall immediately, validate recent "
            "industrial safety predictions, and "
            "consider fallback or human review."
        )

    if high:
        return (
            "Significant model drift detected. "
            "Investigate data drift, feature changes, "
            "label quality, sensor changes, and model "
            "performance. Schedule validation and "
            "retraining if degradation persists."
        )

    if any(
        metric.drift_detected
        for metric in result
    ):
        return (
            "Moderate model drift detected. "
            "Increase monitoring frequency and "
            "compare performance across assets, "
            "sites, model versions, and recent "
            "operating conditions."
        )

    return (
        "Model performance remains within "
        "configured drift thresholds."
    )


# =========================================================
# Main evaluator
# =========================================================

def evaluate_model_drift(
    *,
    model_name: str,
    model_version: str,
    task_type: ModelTaskType,

    baseline_observations: Sequence[
        ModelObservation
    ],

    current_observations: Sequence[
        ModelObservation
    ],

    config: ModelDriftConfig | None = None,

    positive_label: Any = 1,

    metadata: dict[str, Any] | None = None,
) -> ModelDriftResult:

    config = (
        config
        or ModelDriftConfig()
    )

    baseline_count = len(
        baseline_observations
    )

    current_count = len(
        current_observations
    )

    if (
        baseline_count
        < config.minimum_samples
        or current_count
        < config.minimum_samples
    ):
        return ModelDriftResult(
            model_name=model_name,
            model_version=model_version,
            task_type=task_type,

            baseline_samples=baseline_count,
            current_samples=current_count,

            status=(
                DriftStatus
                .INSUFFICIENT_DATA
            ),

            severity=DriftSeverity.NONE,

            drift_detected=False,

            overall_drift_score=0.0,

            drifted_metrics=[],
            critical_metrics=[],

            metric_results=[],

            baseline_metrics={},
            current_metrics={},

            recommendation=(
                "Insufficient labeled samples "
                "for reliable model drift "
                "evaluation."
            ),

            metadata=metadata or {},
        )

    baseline_snapshots = (
        build_metric_snapshots(
            baseline_observations,
            positive_label=positive_label,
        )
    )

    current_snapshots = (
        build_metric_snapshots(
            current_observations,
            positive_label=positive_label,
        )
    )

    comparable_metrics = (
        set(
            baseline_snapshots
        )
        & set(
            current_snapshots
        )
    )

    metric_results: list[
        MetricDriftResult
    ] = []

    for metric in sorted(
        comparable_metrics,
        key=lambda item: item.value,
    ):
        metric_results.append(
            compare_metric(
                baseline=(
                    baseline_snapshots[
                        metric
                    ]
                ),

                current=(
                    current_snapshots[
                        metric
                    ]
                ),

                config=config,
            )
        )

    severity = highest_severity(
        metric_results
    )

    drifted = [
        result.metric.value
        for result in metric_results
        if result.drift_detected
    ]

    critical = [
        result.metric.value
        for result in metric_results
        if (
            result.severity
            == DriftSeverity.CRITICAL
        )
    ]

    weighted_scores: list[
        float
    ] = []

    for result in metric_results:
        weighted_scores.append(
            result.drift_score
        )

    overall_score = (
        mean(weighted_scores)
        if weighted_scores
        else 0.0
    )

    return ModelDriftResult(
        model_name=model_name,
        model_version=model_version,
        task_type=task_type,

        baseline_samples=baseline_count,
        current_samples=current_count,

        status=determine_status(
            severity
        ),

        severity=severity,

        drift_detected=bool(
            drifted
        ),

        overall_drift_score=(
            overall_score
        ),

        drifted_metrics=drifted,

        critical_metrics=critical,

        metric_results=metric_results,

        baseline_metrics={
            metric.value:
                snapshot.value
            for metric, snapshot
            in baseline_snapshots.items()
        },

        current_metrics={
            metric.value:
                snapshot.value
            for metric, snapshot
            in current_snapshots.items()
        },

        recommendation=(
            build_recommendation(
                metric_results
            )
        ),

        metadata=metadata or {},
    )


# =========================================================
# Per-asset drift
# =========================================================

def evaluate_drift_by_asset(
    observations: Sequence[
        ModelObservation
    ],
    *,
    baseline_start: datetime,
    baseline_end: datetime,

    current_start: datetime,
    current_end: datetime,

    config: ModelDriftConfig | None = None,

    positive_label: Any = 1,
) -> dict[str, ModelDriftResult]:

    grouped: dict[
        str,
        list[ModelObservation],
    ] = defaultdict(list)

    for observation in observations:
        if observation.asset_id:
            grouped[
                observation.asset_id
            ].append(
                observation
            )

    results: dict[
        str,
        ModelDriftResult,
    ] = {}

    for asset_id, records in grouped.items():

        baseline = [
            record
            for record in records
            if (
                baseline_start
                <= record.timestamp
                < baseline_end
            )
        ]

        current = [
            record
            for record in records
            if (
                current_start
                <= record.timestamp
                < current_end
            )
        ]

        if not records:
            continue

        first = records[0]

        results[
            asset_id
        ] = evaluate_model_drift(
            model_name=(
                first.model_name
            ),

            model_version=(
                first.model_version
            ),

            task_type=(
                first.task_type
            ),

            baseline_observations=(
                baseline
            ),

            current_observations=(
                current
            ),

            config=config,

            positive_label=(
                positive_label
            ),

            metadata={
                "asset_id":
                    asset_id,
            },
        )

    return results


# =========================================================
# Per-site drift
# =========================================================

def evaluate_drift_by_site(
    observations: Sequence[
        ModelObservation
    ],
    *,
    baseline_start: datetime,
    baseline_end: datetime,

    current_start: datetime,
    current_end: datetime,

    config: ModelDriftConfig | None = None,

    positive_label: Any = 1,
) -> dict[str, ModelDriftResult]:

    grouped: dict[
        str,
        list[ModelObservation],
    ] = defaultdict(list)

    for observation in observations:
        if observation.site_id:
            grouped[
                observation.site_id
            ].append(
                observation
            )

    results: dict[
        str,
        ModelDriftResult,
    ] = {}

    for site_id, records in grouped.items():

        baseline = [
            record
            for record in records
            if (
                baseline_start
                <= record.timestamp
                < baseline_end
            )
        ]

        current = [
            record
            for record in records
            if (
                current_start
                <= record.timestamp
                < current_end
            )
        ]

        first = records[0]

        results[
            site_id
        ] = evaluate_model_drift(
            model_name=(
                first.model_name
            ),

            model_version=(
                first.model_version
            ),

            task_type=(
                first.task_type
            ),

            baseline_observations=(
                baseline
            ),

            current_observations=(
                current
            ),

            config=config,

            positive_label=(
                positive_label
            ),

            metadata={
                "site_id":
                    site_id,
            },
        )

    return results


# =========================================================
# Drift history
# =========================================================

@dataclass(slots=True)
class DriftHistory:
    model_name: str

    evaluations: list[
        ModelDriftResult
    ] = field(
        default_factory=list
    )

    def add(
        self,
        result: ModelDriftResult,
    ) -> None:
        self.evaluations.append(
            result
        )

    def latest(
        self,
    ) -> ModelDriftResult | None:
        if not self.evaluations:
            return None

        return self.evaluations[-1]

    def drift_rate(self) -> float:
        if not self.evaluations:
            return 0.0

        drifted = sum(
            result.drift_detected
            for result
            in self.evaluations
        )

        return (
            drifted
            / len(
                self.evaluations
            )
        )

    def critical_count(self) -> int:
        return sum(
            result.severity
            == DriftSeverity.CRITICAL
            for result
            in self.evaluations
        )


# =========================================================
# Monitoring service
# =========================================================

class ModelDriftMonitor:
    def __init__(
        self,
        *,
        config: ModelDriftConfig | None = None,
    ) -> None:

        self.config = (
            config
            or ModelDriftConfig()
        )

        self._history: dict[
            str,
            DriftHistory,
        ] = {}

    def evaluate(
        self,
        *,
        model_name: str,
        model_version: str,
        task_type: ModelTaskType,

        baseline_observations: Sequence[
            ModelObservation
        ],

        current_observations: Sequence[
            ModelObservation
        ],

        positive_label: Any = 1,

        metadata: dict[
            str,
            Any,
        ] | None = None,
    ) -> ModelDriftResult:

        result = evaluate_model_drift(
            model_name=model_name,
            model_version=model_version,
            task_type=task_type,

            baseline_observations=(
                baseline_observations
            ),

            current_observations=(
                current_observations
            ),

            config=self.config,

            positive_label=(
                positive_label
            ),

            metadata=metadata,
        )

        history = (
            self._history.setdefault(
                model_name,
                DriftHistory(
                    model_name=model_name
                ),
            )
        )

        history.add(
            result
        )

        return result

    def history(
        self,
        model_name: str,
    ) -> DriftHistory | None:

        return self._history.get(
            model_name
        )

    def health_summary(
        self,
    ) -> dict[str, Any]:

        models: dict[
            str,
            Any,
        ] = {}

        healthy = 0
        watch = 0
        drifted = 0
        critical = 0

        for (
            model_name,
            history,
        ) in self._history.items():

            latest = history.latest()

            if latest is None:
                continue

            models[
                model_name
            ] = {
                "version":
                    latest.model_version,

                "status":
                    latest.status.value,

                "severity":
                    latest.severity.value,

                "drift_score":
                    latest
                    .overall_drift_score,

                "drifted_metrics":
                    latest
                    .drifted_metrics,

                "evaluated_at":
                    latest
                    .evaluated_at
                    .isoformat(),
            }

            if (
                latest.status
                == DriftStatus.HEALTHY
            ):
                healthy += 1

            elif (
                latest.status
                == DriftStatus.WATCH
            ):
                watch += 1

            elif (
                latest.status
                == DriftStatus.CRITICAL
            ):
                critical += 1

            elif (
                latest.status
                == DriftStatus.DRIFTED
            ):
                drifted += 1

        return {
            "models_monitored":
                len(models),

            "healthy":
                healthy,

            "watch":
                watch,

            "drifted":
                drifted,

            "critical":
                critical,

            "models":
                models,
        }


# =========================================================
# Observability integration
# =========================================================

def publish_drift_metrics(
    result: ModelDriftResult,
) -> None:
    """
    Optional bridge to app.observability.metrics.

    Import is intentionally local to avoid creating
    hard circular dependencies.
    """

    try:
        from app.observability.metrics import (
            record_data_drift,
        )

    except ImportError:
        return

    record_data_drift(
        model_name=(
            result.model_name
        ),

        model_version=(
            result.model_version
        ),

        score=(
            result.overall_drift_score
        ),

        feature_ratio=(
            len(
                result.drifted_metrics
            )
            / max(
                len(
                    result.metric_results
                ),
                1,
            )
        ),

        drift_detected=(
            result.drift_detected
        ),

        severity=(
            result.severity.value
        ),
    )


def log_model_drift(
    result: ModelDriftResult,
) -> None:
    """
    Optional bridge to structured logging.
    """

    try:
        from app.observability.logging import (
            get_logger,
            LogEventType,
        )

    except ImportError:
        return

    logger = get_logger(
        "industrial.model_drift"
    )

    data = {
        "model_name":
            result.model_name,

        "model_version":
            result.model_version,

        "task_type":
            result.task_type.value,

        "status":
            result.status.value,

        "severity":
            result.severity.value,

        "drift_detected":
            result.drift_detected,

        "overall_drift_score":
            result.overall_drift_score,

        "drifted_metrics":
            result.drifted_metrics,

        "critical_metrics":
            result.critical_metrics,

        "baseline_samples":
            result.baseline_samples,

        "current_samples":
            result.current_samples,
    }

    if result.severity in {
        DriftSeverity.CRITICAL,
        DriftSeverity.HIGH,
    }:
        logger.error(
            "Model performance drift detected.",
            event_type=(
                LogEventType.MODEL_DRIFT
            ),
            data=data,
        )

    elif result.drift_detected:
        logger.warning(
            "Model performance drift detected.",
            event_type=(
                LogEventType.MODEL_DRIFT
            ),
            data=data,
        )

    else:
        logger.info(
            "Model drift evaluation completed.",
            event_type=(
                LogEventType.MODEL_DRIFT
            ),
            data=data,
        )


# =========================================================
# Global monitor
# =========================================================

model_drift_monitor = ModelDriftMonitor()


__all__ = [
    "ModelTaskType",
    "DriftSeverity",
    "DriftStatus",
    "DriftMetric",
    "MetricDirection",

    "DriftThreshold",
    "ModelDriftConfig",

    "ModelObservation",
    "ModelMetricSnapshot",

    "MetricDriftResult",
    "ModelDriftResult",

    "BinaryClassificationMetrics",
    "RegressionMetrics",

    "calculate_binary_metrics",
    "calculate_regression_metrics",
    "calculate_multiclass_accuracy",
    "calculate_average_latency",

    "build_metric_snapshots",

    "calculate_degradation",
    "classify_drift_severity",
    "compare_metric",

    "highest_severity",
    "determine_status",

    "evaluate_model_drift",
    "evaluate_drift_by_asset",
    "evaluate_drift_by_site",

    "DriftHistory",
    "ModelDriftMonitor",

    "publish_drift_metrics",
    "log_model_drift",

    "model_drift_monitor",
]