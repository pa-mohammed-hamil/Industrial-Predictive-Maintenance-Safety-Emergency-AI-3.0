# app/evaluation/anomaly_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import ceil, isfinite, sqrt
from statistics import mean
from typing import Any, Iterable


# =========================================================
# Enums
# =========================================================

class AnomalyType(str, Enum):
    NORMAL = "normal"

    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    ENERGY = "energy"
    ELECTRICAL = "electrical"

    MULTIVARIATE = "multivariate"
    SENSOR_FAULT = "sensor_fault"

    SPIKE = "spike"
    DROP = "drop"
    DRIFT = "drift"
    OSCILLATION = "oscillation"
    OUTLIER = "outlier"

    UNKNOWN = "unknown"


class AnomalySeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =========================================================
# Prediction / ground truth
# =========================================================

@dataclass(slots=True)
class AnomalyPrediction:
    is_anomaly: bool

    anomaly_type: AnomalyType = (
        AnomalyType.NORMAL
    )

    anomaly_score: float = 0.0
    confidence: float = 1.0

    severity: AnomalySeverity = (
        AnomalySeverity.NONE
    )

    detection_timestamp_ms: float | None = None

    model_name: str | None = None
    model_version: str | None = None

    contributing_features: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class AnomalyGroundTruth:
    is_anomaly: bool

    anomaly_type: AnomalyType = (
        AnomalyType.NORMAL
    )

    severity: AnomalySeverity = (
        AnomalySeverity.NONE
    )

    event_timestamp_ms: float | None = None


@dataclass(slots=True)
class AnomalyEvaluationSample:
    sample_id: str
    asset_id: str

    prediction: AnomalyPrediction
    ground_truth: AnomalyGroundTruth

    sensor_id: str | None = None
    site_id: str | None = None

    timestamp_ms: float | None = None

    prediction_id: str | None = None
    incident_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Metrics
# =========================================================

@dataclass(slots=True)
class BinaryAnomalyMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float

    false_positive_rate: float
    false_negative_rate: float

    balanced_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnomalyClassMetrics:
    anomaly_type: str

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    precision: float
    recall: float
    specificity: float
    f1_score: float

    support: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnomalyEvaluationResult:
    total_samples: int

    anomaly_samples: int
    normal_samples: int

    detected_anomalies: int
    missed_anomalies: int
    false_alarms: int

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    balanced_accuracy: float

    false_positive_rate: float
    false_negative_rate: float

    type_accuracy: float | None
    severity_accuracy: float | None

    critical_anomaly_recall: float | None

    macro_precision: float
    macro_recall: float
    macro_f1: float

    average_anomaly_score: float
    average_confidence: float

    expected_calibration_error: float

    average_detection_delay_ms: float | None
    p95_detection_delay_ms: float | None
    max_detection_delay_ms: float | None

    score_separation: float | None

    confusion_matrix: dict[
        str,
        dict[str, int],
    ]

    per_type: dict[
        str,
        AnomalyClassMetrics,
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_type"] = {
            name: metrics.to_dict()
            for name, metrics
            in self.per_type.items()
        }

        return result


# =========================================================
# Validation
# =========================================================

def _validate_probability(
    value: float,
    name: str,
) -> float:
    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def _validate_score(
    value: float,
) -> float:
    value = float(value)

    if not isfinite(value):
        raise ValueError(
            "anomaly_score must be finite."
        )

    return value


# =========================================================
# Binary anomaly metrics
# =========================================================

def calculate_binary_metrics(
    samples: Iterable[
        AnomalyEvaluationSample
    ],
) -> BinaryAnomalyMetrics:

    tp = tn = fp = fn = 0

    for sample in samples:
        actual = (
            sample.ground_truth.is_anomaly
        )

        predicted = (
            sample.prediction.is_anomaly
        )

        if actual and predicted:
            tp += 1

        elif (
            not actual
            and not predicted
        ):
            tn += 1

        elif (
            not actual
            and predicted
        ):
            fp += 1

        else:
            fn += 1

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

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    f1 = (
        2 * precision * recall
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

    balanced_accuracy = (
        recall + specificity
    ) / 2

    return BinaryAnomalyMetrics(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,

        accuracy=round(
            accuracy,
            4,
        ),

        precision=round(
            precision,
            4,
        ),

        recall=round(
            recall,
            4,
        ),

        specificity=round(
            specificity,
            4,
        ),

        f1_score=round(
            f1,
            4,
        ),

        false_positive_rate=round(
            fpr,
            4,
        ),

        false_negative_rate=round(
            fnr,
            4,
        ),

        balanced_accuracy=round(
            balanced_accuracy,
            4,
        ),
    )


# =========================================================
# Anomaly type confusion matrix
# =========================================================

def build_type_confusion_matrix(
    samples: list[
        AnomalyEvaluationSample
    ],
) -> dict[str, dict[str, int]]:

    labels = [
        item.value
        for item in AnomalyType
    ]

    matrix = {
        actual: {
            predicted: 0
            for predicted in labels
        }
        for actual in labels
    }

    for sample in samples:
        actual = (
            sample.ground_truth
            .anomaly_type.value
        )

        predicted = (
            sample.prediction
            .anomaly_type.value
        )

        matrix[
            actual
        ][
            predicted
        ] += 1

    return matrix


# =========================================================
# Per-type metrics
# =========================================================

def calculate_type_metrics(
    *,
    anomaly_type: AnomalyType,
    matrix: dict[
        str,
        dict[str, int],
    ],
) -> AnomalyClassMetrics:

    label = anomaly_type.value

    total = sum(
        sum(row.values())
        for row in matrix.values()
    )

    tp = matrix[label][label]

    fp = sum(
        matrix[actual][label]
        for actual in matrix
        if actual != label
    )

    fn = sum(
        matrix[label][predicted]
        for predicted in matrix[label]
        if predicted != label
    )

    tn = (
        total
        - tp
        - fp
        - fn
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

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return AnomalyClassMetrics(
        anomaly_type=label,

        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,

        precision=round(
            precision,
            4,
        ),

        recall=round(
            recall,
            4,
        ),

        specificity=round(
            specificity,
            4,
        ),

        f1_score=round(
            f1,
            4,
        ),

        support=(
            tp + fn
        ),
    )


# =========================================================
# Anomaly type accuracy
# =========================================================

def calculate_type_accuracy(
    samples: list[
        AnomalyEvaluationSample
    ],
) -> float | None:

    anomaly_samples = [
        sample
        for sample in samples
        if sample.ground_truth.is_anomaly
    ]

    if not anomaly_samples:
        return None

    correct = sum(
        (
            sample.prediction.is_anomaly
            and
            sample.prediction.anomaly_type
            == sample.ground_truth.anomaly_type
        )
        for sample in anomaly_samples
    )

    return round(
        correct / len(anomaly_samples),
        4,
    )


# =========================================================
# Severity accuracy
# =========================================================

def calculate_severity_accuracy(
    samples: list[
        AnomalyEvaluationSample
    ],
) -> float | None:

    anomaly_samples = [
        sample
        for sample in samples
        if sample.ground_truth.is_anomaly
    ]

    if not anomaly_samples:
        return None

    correct = sum(
        (
            sample.prediction.is_anomaly
            and
            sample.prediction.severity
            == sample.ground_truth.severity
        )
        for sample in anomaly_samples
    )

    return round(
        correct / len(anomaly_samples),
        4,
    )


# =========================================================
# Critical anomaly recall
# =========================================================

def calculate_critical_anomaly_recall(
    samples: list[
        AnomalyEvaluationSample
    ],
) -> float | None:

    critical = [
        sample
        for sample in samples
        if (
            sample.ground_truth.is_anomaly
            and
            sample.ground_truth.severity
            == AnomalySeverity.CRITICAL
        )
    ]

    if not critical:
        return None

    detected = sum(
        sample.prediction.is_anomaly
        for sample in critical
    )

    return round(
        detected / len(critical),
        4,
    )


# =========================================================
# Detection delay
# =========================================================

def calculate_detection_delay(
    sample: AnomalyEvaluationSample,
) -> float | None:

    event_time = (
        sample.ground_truth
        .event_timestamp_ms
    )

    detection_time = (
        sample.prediction
        .detection_timestamp_ms
    )

    if (
        event_time is None
        or detection_time is None
    ):
        return None

    return max(
        0.0,
        float(detection_time)
        - float(event_time),
    )


def calculate_detection_delay_metrics(
    samples: list[
        AnomalyEvaluationSample
    ],
) -> tuple[
    float | None,
    float | None,
    float | None,
]:

    delays = sorted(
        delay
        for sample in samples
        if (
            sample.ground_truth.is_anomaly
            and
            sample.prediction.is_anomaly
        )
        if (
            delay :=
            calculate_detection_delay(
                sample
            )
        ) is not None
    )

    if not delays:
        return (
            None,
            None,
            None,
        )

    average_delay = round(
        mean(delays),
        2,
    )

    rank = max(
        1,
        ceil(
            0.95
            * len(delays)
        ),
    )

    index = min(
        rank - 1,
        len(delays) - 1,
    )

    p95 = round(
        delays[index],
        2,
    )

    maximum = round(
        max(delays),
        2,
    )

    return (
        average_delay,
        p95,
        maximum,
    )


# =========================================================
# Confidence calibration
# =========================================================

def calculate_ece(
    samples: list[
        AnomalyEvaluationSample
    ],
    *,
    bins: int = 10,
) -> float:

    if not samples:
        return 0.0

    if bins < 1:
        raise ValueError(
            "bins must be >= 1."
        )

    buckets: list[
        list[
            AnomalyEvaluationSample
        ]
    ] = [
        []
        for _ in range(bins)
    ]

    for sample in samples:
        confidence = (
            _validate_probability(
                sample.prediction.confidence,
                "confidence",
            )
        )

        index = min(
            int(
                confidence
                * bins
            ),
            bins - 1,
        )

        buckets[index].append(
            sample
        )

    ece = 0.0

    for bucket in buckets:
        if not bucket:
            continue

        bucket_accuracy = mean(
            1.0
            if (
                sample.prediction.is_anomaly
                == sample.ground_truth.is_anomaly
            )
            else 0.0
            for sample in bucket
        )

        bucket_confidence = mean(
            sample.prediction.confidence
            for sample in bucket
        )

        ece += (
            len(bucket)
            / len(samples)
        ) * abs(
            bucket_accuracy
            - bucket_confidence
        )

    return round(
        ece,
        4,
    )


# =========================================================
# Score separation
# =========================================================

def calculate_score_separation(
    samples: list[
        AnomalyEvaluationSample
    ],
) -> float | None:
    """
    Measures separation between anomaly scores for
    actual anomalies and actual normal samples.

    Positive and larger is generally better when higher
    anomaly_score means "more anomalous".
    """

    anomaly_scores = [
        sample.prediction.anomaly_score
        for sample in samples
        if sample.ground_truth.is_anomaly
    ]

    normal_scores = [
        sample.prediction.anomaly_score
        for sample in samples
        if not sample.ground_truth.is_anomaly
    ]

    if (
        not anomaly_scores
        or not normal_scores
    ):
        return None

    separation = (
        mean(anomaly_scores)
        - mean(normal_scores)
    )

    return round(
        separation,
        6,
    )


# =========================================================
# RMSE utility for anomaly scores
# =========================================================

def calculate_score_rmse(
    predicted_scores: list[float],
    target_scores: list[float],
) -> float:

    if (
        len(predicted_scores)
        != len(target_scores)
    ):
        raise ValueError(
            "Score arrays must have "
            "equal lengths."
        )

    if not predicted_scores:
        return 0.0

    squared_errors = [
        (
            float(predicted)
            - float(target)
        ) ** 2
        for predicted, target
        in zip(
            predicted_scores,
            target_scores,
        )
    ]

    return round(
        sqrt(
            mean(squared_errors)
        ),
        6,
    )


# =========================================================
# Threshold evaluation
# =========================================================

def evaluate_anomaly_threshold(
    samples: list[
        AnomalyEvaluationSample
    ],
    *,
    threshold: float,
) -> dict[str, Any]:
    """
    Re-evaluate anomaly decisions directly from the
    model's anomaly_score.

    Assumption:
        anomaly_score >= threshold => anomaly

    This is useful for threshold tuning before deployment.
    """

    threshold = float(threshold)

    if not isfinite(threshold):
        raise ValueError(
            "threshold must be finite."
        )

    tp = tn = fp = fn = 0

    for sample in samples:
        score = _validate_score(
            sample.prediction.anomaly_score
        )

        predicted = (
            score >= threshold
        )

        actual = (
            sample.ground_truth.is_anomaly
        )

        if actual and predicted:
            tp += 1

        elif (
            not actual
            and not predicted
        ):
            tn += 1

        elif (
            not actual
            and predicted
        ):
            fp += 1

        else:
            fn += 1

    total = (
        tp + tn + fp + fn
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
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    return {
        "threshold":
            threshold,

        "samples":
            total,

        "true_positive":
            tp,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "precision": round(
            precision,
            4,
        ),

        "recall": round(
            recall,
            4,
        ),

        "f1_score": round(
            f1,
            4,
        ),

        "false_positive_rate": round(
            fpr,
            4,
        ),
    }


# =========================================================
# Find best threshold
# =========================================================

def find_best_threshold(
    samples: list[
        AnomalyEvaluationSample
    ],
    *,
    thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:

    if not samples:
        return {
            "threshold": None,
            "f1_score": 0.0,
        }

    if thresholds is None:
        scores = sorted({
            float(
                sample.prediction.anomaly_score
            )
            for sample in samples
        })

        thresholds = scores

    best: dict[str, Any] | None = None

    for threshold in thresholds:
        result = (
            evaluate_anomaly_threshold(
                samples,
                threshold=float(
                    threshold
                ),
            )
        )

        if (
            best is None
            or result["f1_score"]
            > best["f1_score"]
        ):
            best = result

    return best or {
        "threshold": None,
        "f1_score": 0.0,
    }


# =========================================================
# Main evaluator
# =========================================================

def evaluate_anomalies(
    samples: list[
        AnomalyEvaluationSample
    ],
) -> AnomalyEvaluationResult:

    if not samples:
        return AnomalyEvaluationResult(
            total_samples=0,

            anomaly_samples=0,
            normal_samples=0,

            detected_anomalies=0,
            missed_anomalies=0,
            false_alarms=0,

            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            specificity=0.0,
            f1_score=0.0,
            balanced_accuracy=0.0,

            false_positive_rate=0.0,
            false_negative_rate=0.0,

            type_accuracy=None,
            severity_accuracy=None,

            critical_anomaly_recall=None,

            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,

            average_anomaly_score=0.0,
            average_confidence=0.0,

            expected_calibration_error=0.0,

            average_detection_delay_ms=None,
            p95_detection_delay_ms=None,
            max_detection_delay_ms=None,

            score_separation=None,

            confusion_matrix={},
            per_type={},
            errors=[],
        )

    for sample in samples:
        _validate_probability(
            sample.prediction.confidence,
            "confidence",
        )

        _validate_score(
            sample.prediction.anomaly_score
        )

    binary = calculate_binary_metrics(
        samples
    )

    matrix = build_type_confusion_matrix(
        samples
    )

    per_type = {
        anomaly_type.value:
            calculate_type_metrics(
                anomaly_type=anomaly_type,
                matrix=matrix,
            )
        for anomaly_type
        in AnomalyType
    }

    supported = [
        metric
        for metric in per_type.values()
        if metric.support > 0
    ]

    if supported:
        macro_precision = mean(
            metric.precision
            for metric in supported
        )

        macro_recall = mean(
            metric.recall
            for metric in supported
        )

        macro_f1 = mean(
            metric.f1_score
            for metric in supported
        )

    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    (
        average_delay,
        p95_delay,
        max_delay,
    ) = calculate_detection_delay_metrics(
        samples
    )

    anomaly_samples = sum(
        sample.ground_truth.is_anomaly
        for sample in samples
    )

    normal_samples = (
        len(samples)
        - anomaly_samples
    )

    errors: list[
        dict[str, Any]
    ] = []

    for sample in samples:
        detection_correct = (
            sample.prediction.is_anomaly
            == sample.ground_truth.is_anomaly
        )

        type_correct = (
            sample.prediction.anomaly_type
            == sample.ground_truth.anomaly_type
        )

        severity_correct = (
            sample.prediction.severity
            == sample.ground_truth.severity
        )

        if (
            not detection_correct
            or (
                sample.ground_truth.is_anomaly
                and (
                    not type_correct
                    or not severity_correct
                )
            )
        ):
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "asset_id":
                        sample.asset_id,

                    "sensor_id":
                        sample.sensor_id,

                    "site_id":
                        sample.site_id,

                    "actual_is_anomaly":
                        sample.ground_truth
                        .is_anomaly,

                    "predicted_is_anomaly":
                        sample.prediction
                        .is_anomaly,

                    "actual_type":
                        sample.ground_truth
                        .anomaly_type.value,

                    "predicted_type":
                        sample.prediction
                        .anomaly_type.value,

                    "actual_severity":
                        sample.ground_truth
                        .severity.value,

                    "predicted_severity":
                        sample.prediction
                        .severity.value,

                    "anomaly_score":
                        sample.prediction
                        .anomaly_score,

                    "confidence":
                        sample.prediction
                        .confidence,

                    "detection_delay_ms":
                        calculate_detection_delay(
                            sample
                        ),

                    "prediction_id":
                        sample.prediction_id,

                    "incident_id":
                        sample.incident_id,
                }
            )

    return AnomalyEvaluationResult(
        total_samples=len(samples),

        anomaly_samples=anomaly_samples,
        normal_samples=normal_samples,

        detected_anomalies=(
            binary.true_positive
        ),

        missed_anomalies=(
            binary.false_negative
        ),

        false_alarms=(
            binary.false_positive
        ),

        accuracy=binary.accuracy,
        precision=binary.precision,
        recall=binary.recall,
        specificity=binary.specificity,
        f1_score=binary.f1_score,

        balanced_accuracy=(
            binary.balanced_accuracy
        ),

        false_positive_rate=(
            binary.false_positive_rate
        ),

        false_negative_rate=(
            binary.false_negative_rate
        ),

        type_accuracy=(
            calculate_type_accuracy(
                samples
            )
        ),

        severity_accuracy=(
            calculate_severity_accuracy(
                samples
            )
        ),

        critical_anomaly_recall=(
            calculate_critical_anomaly_recall(
                samples
            )
        ),

        macro_precision=round(
            macro_precision,
            4,
        ),

        macro_recall=round(
            macro_recall,
            4,
        ),

        macro_f1=round(
            macro_f1,
            4,
        ),

        average_anomaly_score=round(
            mean(
                sample.prediction.anomaly_score
                for sample in samples
            ),
            6,
        ),

        average_confidence=round(
            mean(
                sample.prediction.confidence
                for sample in samples
            ),
            4,
        ),

        expected_calibration_error=(
            calculate_ece(
                samples
            )
        ),

        average_detection_delay_ms=(
            average_delay
        ),

        p95_detection_delay_ms=(
            p95_delay
        ),

        max_detection_delay_ms=(
            max_delay
        ),

        score_separation=(
            calculate_score_separation(
                samples
            )
        ),

        confusion_matrix=matrix,
        per_type=per_type,
        errors=errors,
    )


# =========================================================
# Evaluator service
# =========================================================

class AnomalyEvaluator:

    def evaluate(
        self,
        samples: list[
            AnomalyEvaluationSample
        ],
    ) -> AnomalyEvaluationResult:

        return evaluate_anomalies(
            samples
        )

    def evaluate_threshold(
        self,
        samples: list[
            AnomalyEvaluationSample
        ],
        *,
        threshold: float,
    ) -> dict[str, Any]:

        return evaluate_anomaly_threshold(
            samples,
            threshold=threshold,
        )

    def best_threshold(
        self,
        samples: list[
            AnomalyEvaluationSample
        ],
        *,
        thresholds: Iterable[
            float
        ] | None = None,
    ) -> dict[str, Any]:

        return find_best_threshold(
            samples,
            thresholds=thresholds,
        )

    def critical_recall(
        self,
        samples: list[
            AnomalyEvaluationSample
        ],
    ) -> float | None:

        return (
            calculate_critical_anomaly_recall(
                samples
            )
        )


anomaly_evaluator = AnomalyEvaluator()


__all__ = [
    "AnomalyType",
    "AnomalySeverity",
    "AnomalyPrediction",
    "AnomalyGroundTruth",
    "AnomalyEvaluationSample",
    "BinaryAnomalyMetrics",
    "AnomalyClassMetrics",
    "AnomalyEvaluationResult",
    "calculate_binary_metrics",
    "build_type_confusion_matrix",
    "calculate_type_metrics",
    "calculate_type_accuracy",
    "calculate_severity_accuracy",
    "calculate_critical_anomaly_recall",
    "calculate_detection_delay",
    "calculate_detection_delay_metrics",
    "calculate_ece",
    "calculate_score_separation",
    "calculate_score_rmse",
    "evaluate_anomaly_threshold",
    "find_best_threshold",
    "evaluate_anomalies",
    "AnomalyEvaluator",
    "anomaly_evaluator",
]