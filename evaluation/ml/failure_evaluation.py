# app/evaluation/failure_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import ceil, isfinite
from statistics import mean
from typing import Any, Iterable


# =========================================================
# Enums
# =========================================================

class FailureType(str, Enum):
    NO_FAILURE = "no_failure"

    BEARING_FAILURE = "bearing_failure"
    MOTOR_FAILURE = "motor_failure"
    GEAR_FAILURE = "gear_failure"
    SHAFT_FAILURE = "shaft_failure"
    PUMP_FAILURE = "pump_failure"
    VALVE_FAILURE = "valve_failure"
    ELECTRICAL_FAILURE = "electrical_failure"

    OVERHEATING = "overheating"
    LUBRICATION_FAILURE = "lubrication_failure"
    COOLING_FAILURE = "cooling_failure"

    PRESSURE_FAILURE = "pressure_failure"
    HYDRAULIC_FAILURE = "hydraulic_failure"
    PNEUMATIC_FAILURE = "pneumatic_failure"

    SENSOR_FAILURE = "sensor_failure"
    CONTROL_FAILURE = "control_failure"

    STRUCTURAL_FAILURE = "structural_failure"

    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =========================================================
# Prediction / ground truth
# =========================================================

@dataclass(slots=True)
class FailurePrediction:
    will_fail: bool

    failure_probability: float

    failure_type: FailureType = (
        FailureType.NO_FAILURE
    )

    severity: FailureSeverity = (
        FailureSeverity.NONE
    )

    # Predicted hours until failure.
    predicted_time_to_failure_hours: (
        float | None
    ) = None

    # Timestamp when prediction was generated.
    prediction_timestamp_ms: (
        float | None
    ) = None

    model_name: str | None = None
    model_version: str | None = None

    contributing_features: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class FailureGroundTruth:
    failed: bool

    failure_type: FailureType = (
        FailureType.NO_FAILURE
    )

    severity: FailureSeverity = (
        FailureSeverity.NONE
    )

    # Actual timestamp at which failure occurred.
    failure_timestamp_ms: (
        float | None
    ) = None

    # Optional actual TTF if already calculated.
    actual_time_to_failure_hours: (
        float | None
    ) = None


@dataclass(slots=True)
class FailureEvaluationSample:
    sample_id: str
    asset_id: str

    prediction: FailurePrediction
    ground_truth: FailureGroundTruth

    site_id: str | None = None
    sensor_id: str | None = None

    prediction_id: str | None = None
    work_order_id: str | None = None
    incident_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Binary metrics
# =========================================================

@dataclass(slots=True)
class BinaryFailureMetrics:
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


# =========================================================
# Per-failure-type metrics
# =========================================================

@dataclass(slots=True)
class FailureClassMetrics:
    failure_type: str

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    precision: float
    recall: float
    specificity: float
    f1_score: float

    support: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# Main result
# =========================================================

@dataclass(slots=True)
class FailureEvaluationResult:
    total_samples: int

    failure_samples: int
    healthy_samples: int

    correctly_predicted_failures: int
    missed_failures: int
    false_failure_alarms: int

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float

    balanced_accuracy: float

    false_positive_rate: float
    false_negative_rate: float

    failure_type_accuracy: float | None
    severity_accuracy: float | None

    critical_failure_recall: float | None

    macro_precision: float
    macro_recall: float
    macro_f1: float

    average_failure_probability: float
    expected_calibration_error: float
    brier_score: float

    ttf_mae_hours: float | None

    average_prediction_lead_time_hours: (
        float | None
    )

    p95_prediction_lead_time_hours: (
        float | None
    )

    insufficient_lead_time_rate: (
        float | None
    )

    confusion_matrix: dict[
        str,
        dict[str, int],
    ]

    per_failure_type: dict[
        str,
        FailureClassMetrics,
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_failure_type"] = {
            name: metrics.to_dict()
            for name, metrics
            in self.per_failure_type.items()
        }

        return result


# =========================================================
# Validation
# =========================================================

def _validate_probability(
    value: float,
    name: str = "probability",
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


def _non_negative(
    value: float,
    name: str,
) -> float:

    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


# =========================================================
# Binary failure metrics
# =========================================================

def calculate_binary_metrics(
    samples: Iterable[
        FailureEvaluationSample
    ],
) -> BinaryFailureMetrics:

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in samples:
        actual = (
            sample.ground_truth.failed
        )

        predicted = (
            sample.prediction.will_fail
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
        (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
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

    return BinaryFailureMetrics(
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
# Failure type confusion matrix
# =========================================================

def build_failure_confusion_matrix(
    samples: list[
        FailureEvaluationSample
    ],
) -> dict[str, dict[str, int]]:

    labels = [
        item.value
        for item in FailureType
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
            .failure_type.value
        )

        predicted = (
            sample.prediction
            .failure_type.value
        )

        matrix[
            actual
        ][
            predicted
        ] += 1

    return matrix


# =========================================================
# Failure-type metrics
# =========================================================

def calculate_failure_type_metrics(
    *,
    failure_type: FailureType,
    matrix: dict[
        str,
        dict[str, int],
    ],
) -> FailureClassMetrics:

    label = failure_type.value

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
        (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
        if precision + recall
        else 0.0
    )

    return FailureClassMetrics(
        failure_type=label,

        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,

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
# Failure type accuracy
# =========================================================

def calculate_failure_type_accuracy(
    samples: list[
        FailureEvaluationSample
    ],
) -> float | None:

    failures = [
        sample
        for sample in samples
        if sample.ground_truth.failed
    ]

    if not failures:
        return None

    correct = sum(
        (
            sample.prediction.will_fail
            and
            sample.prediction.failure_type
            == sample.ground_truth.failure_type
        )
        for sample in failures
    )

    return round(
        correct / len(failures),
        4,
    )


# =========================================================
# Severity accuracy
# =========================================================

def calculate_severity_accuracy(
    samples: list[
        FailureEvaluationSample
    ],
) -> float | None:

    failures = [
        sample
        for sample in samples
        if sample.ground_truth.failed
    ]

    if not failures:
        return None

    correct = sum(
        (
            sample.prediction.will_fail
            and
            sample.prediction.severity
            == sample.ground_truth.severity
        )
        for sample in failures
    )

    return round(
        correct / len(failures),
        4,
    )


# =========================================================
# Critical failure recall
# =========================================================

def calculate_critical_failure_recall(
    samples: list[
        FailureEvaluationSample
    ],
) -> float | None:

    critical = [
        sample
        for sample in samples
        if (
            sample.ground_truth.failed
            and
            sample.ground_truth.severity
            == FailureSeverity.CRITICAL
        )
    ]

    if not critical:
        return None

    detected = sum(
        sample.prediction.will_fail
        for sample in critical
    )

    return round(
        detected / len(critical),
        4,
    )


# =========================================================
# Brier score
# =========================================================

def calculate_brier_score(
    samples: list[
        FailureEvaluationSample
    ],
) -> float:

    if not samples:
        return 0.0

    squared_errors: list[float] = []

    for sample in samples:
        probability = (
            _validate_probability(
                sample.prediction
                .failure_probability,
                "failure_probability",
            )
        )

        actual = (
            1.0
            if sample.ground_truth.failed
            else 0.0
        )

        squared_errors.append(
            (
                probability
                - actual
            ) ** 2
        )

    return round(
        mean(squared_errors),
        6,
    )


# =========================================================
# Calibration error
# =========================================================

def calculate_ece(
    samples: list[
        FailureEvaluationSample
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
            FailureEvaluationSample
        ]
    ] = [
        []
        for _ in range(bins)
    ]

    for sample in samples:
        probability = (
            _validate_probability(
                sample.prediction
                .failure_probability,
                "failure_probability",
            )
        )

        index = min(
            int(
                probability
                * bins
            ),
            bins - 1,
        )

        buckets[index].append(
            sample
        )

    error = 0.0

    for bucket in buckets:
        if not bucket:
            continue

        observed_failure_rate = mean(
            1.0
            if sample.ground_truth.failed
            else 0.0
            for sample in bucket
        )

        average_probability = mean(
            sample.prediction
            .failure_probability
            for sample in bucket
        )

        error += (
            len(bucket)
            / len(samples)
        ) * abs(
            observed_failure_rate
            - average_probability
        )

    return round(
        error,
        4,
    )


# =========================================================
# Actual time to failure
# =========================================================

def get_actual_ttf_hours(
    sample: FailureEvaluationSample,
) -> float | None:

    explicit = (
        sample.ground_truth
        .actual_time_to_failure_hours
    )

    if explicit is not None:
        return _non_negative(
            explicit,
            "actual_time_to_failure_hours",
        )

    prediction_time = (
        sample.prediction
        .prediction_timestamp_ms
    )

    failure_time = (
        sample.ground_truth
        .failure_timestamp_ms
    )

    if (
        prediction_time is None
        or failure_time is None
    ):
        return None

    difference_ms = (
        float(failure_time)
        - float(prediction_time)
    )

    if difference_ms < 0:
        return None

    return (
        difference_ms
        / 3_600_000.0
    )


# =========================================================
# TTF accuracy
# =========================================================

def calculate_ttf_mae(
    samples: list[
        FailureEvaluationSample
    ],
) -> float | None:

    errors: list[float] = []

    for sample in samples:
        if not sample.ground_truth.failed:
            continue

        predicted = (
            sample.prediction
            .predicted_time_to_failure_hours
        )

        actual = get_actual_ttf_hours(
            sample
        )

        if (
            predicted is None
            or actual is None
        ):
            continue

        predicted = _non_negative(
            predicted,
            "predicted_time_to_failure_hours",
        )

        errors.append(
            abs(
                predicted
                - actual
            )
        )

    if not errors:
        return None

    return round(
        mean(errors),
        4,
    )


# =========================================================
# Prediction lead time
# =========================================================

def calculate_lead_time_metrics(
    samples: list[
        FailureEvaluationSample
    ],
    *,
    minimum_actionable_hours: float = 24.0,
) -> tuple[
    float | None,
    float | None,
    float | None,
]:

    minimum_actionable_hours = (
        _non_negative(
            minimum_actionable_hours,
            "minimum_actionable_hours",
        )
    )

    lead_times: list[float] = []

    for sample in samples:
        if not (
            sample.ground_truth.failed
            and sample.prediction.will_fail
        ):
            continue

        actual_ttf = get_actual_ttf_hours(
            sample
        )

        if actual_ttf is not None:
            lead_times.append(
                actual_ttf
            )

    if not lead_times:
        return (
            None,
            None,
            None,
        )

    lead_times.sort()

    average_lead_time = round(
        mean(lead_times),
        4,
    )

    rank = max(
        1,
        ceil(
            0.95
            * len(lead_times)
        ),
    )

    index = min(
        rank - 1,
        len(lead_times) - 1,
    )

    p95 = round(
        lead_times[index],
        4,
    )

    insufficient = sum(
        lead_time
        < minimum_actionable_hours
        for lead_time in lead_times
    )

    insufficient_rate = round(
        insufficient
        / len(lead_times),
        4,
    )

    return (
        average_lead_time,
        p95,
        insufficient_rate,
    )


# =========================================================
# Probability threshold evaluation
# =========================================================

def evaluate_failure_threshold(
    samples: list[
        FailureEvaluationSample
    ],
    *,
    threshold: float,
) -> dict[str, Any]:

    threshold = _validate_probability(
        threshold,
        "threshold",
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in samples:
        probability = (
            _validate_probability(
                sample.prediction
                .failure_probability,
                "failure_probability",
            )
        )

        predicted = (
            probability >= threshold
        )

        actual = (
            sample.ground_truth.failed
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
        (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
        if precision + recall
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    return {
        "threshold":
            threshold,

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

        "specificity": round(
            specificity,
            4,
        ),

        "f1_score": round(
            f1,
            4,
        ),
    }


# =========================================================
# Best threshold
# =========================================================

def find_best_failure_threshold(
    samples: list[
        FailureEvaluationSample
    ],
    *,
    thresholds: Iterable[
        float
    ] | None = None,
) -> dict[str, Any]:

    if not samples:
        return {
            "threshold": None,
            "f1_score": 0.0,
        }

    if thresholds is None:
        thresholds = sorted({
            sample.prediction
            .failure_probability
            for sample in samples
        })

    best: dict[str, Any] | None = None

    for threshold in thresholds:
        result = (
            evaluate_failure_threshold(
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
# Main evaluation
# =========================================================

def evaluate_failures(
    samples: list[
        FailureEvaluationSample
    ],
    *,
    minimum_actionable_hours: float = 24.0,
) -> FailureEvaluationResult:

    if not samples:
        return FailureEvaluationResult(
            total_samples=0,

            failure_samples=0,
            healthy_samples=0,

            correctly_predicted_failures=0,
            missed_failures=0,
            false_failure_alarms=0,

            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            specificity=0.0,
            f1_score=0.0,
            balanced_accuracy=0.0,

            false_positive_rate=0.0,
            false_negative_rate=0.0,

            failure_type_accuracy=None,
            severity_accuracy=None,

            critical_failure_recall=None,

            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,

            average_failure_probability=0.0,
            expected_calibration_error=0.0,
            brier_score=0.0,

            ttf_mae_hours=None,

            average_prediction_lead_time_hours=None,
            p95_prediction_lead_time_hours=None,
            insufficient_lead_time_rate=None,

            confusion_matrix={},
            per_failure_type={},
            errors=[],
        )

    for sample in samples:
        _validate_probability(
            sample.prediction
            .failure_probability,
            "failure_probability",
        )

    binary = calculate_binary_metrics(
        samples
    )

    matrix = (
        build_failure_confusion_matrix(
            samples
        )
    )

    per_failure_type = {
        failure_type.value:
            calculate_failure_type_metrics(
                failure_type=failure_type,
                matrix=matrix,
            )
        for failure_type in FailureType
    }

    supported = [
        metric
        for metric
        in per_failure_type.values()
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
        average_lead,
        p95_lead,
        insufficient_lead_rate,
    ) = calculate_lead_time_metrics(
        samples,
        minimum_actionable_hours=(
            minimum_actionable_hours
        ),
    )

    failure_samples = sum(
        sample.ground_truth.failed
        for sample in samples
    )

    healthy_samples = (
        len(samples)
        - failure_samples
    )

    errors: list[
        dict[str, Any]
    ] = []

    for sample in samples:
        detection_correct = (
            sample.prediction.will_fail
            == sample.ground_truth.failed
        )

        type_correct = (
            sample.prediction.failure_type
            == sample.ground_truth.failure_type
        )

        severity_correct = (
            sample.prediction.severity
            == sample.ground_truth.severity
        )

        if (
            not detection_correct
            or (
                sample.ground_truth.failed
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

                    "site_id":
                        sample.site_id,

                    "actual_failure":
                        sample.ground_truth.failed,

                    "predicted_failure":
                        sample.prediction.will_fail,

                    "failure_probability":
                        sample.prediction
                        .failure_probability,

                    "actual_failure_type":
                        sample.ground_truth
                        .failure_type.value,

                    "predicted_failure_type":
                        sample.prediction
                        .failure_type.value,

                    "actual_severity":
                        sample.ground_truth
                        .severity.value,

                    "predicted_severity":
                        sample.prediction
                        .severity.value,

                    "predicted_ttf_hours":
                        sample.prediction
                        .predicted_time_to_failure_hours,

                    "actual_ttf_hours":
                        get_actual_ttf_hours(
                            sample
                        ),

                    "prediction_id":
                        sample.prediction_id,

                    "work_order_id":
                        sample.work_order_id,

                    "incident_id":
                        sample.incident_id,
                }
            )

    return FailureEvaluationResult(
        total_samples=len(samples),

        failure_samples=failure_samples,
        healthy_samples=healthy_samples,

        correctly_predicted_failures=(
            binary.true_positive
        ),

        missed_failures=(
            binary.false_negative
        ),

        false_failure_alarms=(
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

        failure_type_accuracy=(
            calculate_failure_type_accuracy(
                samples
            )
        ),

        severity_accuracy=(
            calculate_severity_accuracy(
                samples
            )
        ),

        critical_failure_recall=(
            calculate_critical_failure_recall(
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

        average_failure_probability=round(
            mean(
                sample.prediction
                .failure_probability
                for sample in samples
            ),
            4,
        ),

        expected_calibration_error=(
            calculate_ece(
                samples
            )
        ),

        brier_score=(
            calculate_brier_score(
                samples
            )
        ),

        ttf_mae_hours=(
            calculate_ttf_mae(
                samples
            )
        ),

        average_prediction_lead_time_hours=(
            average_lead
        ),

        p95_prediction_lead_time_hours=(
            p95_lead
        ),

        insufficient_lead_time_rate=(
            insufficient_lead_rate
        ),

        confusion_matrix=matrix,

        per_failure_type=(
            per_failure_type
        ),

        errors=errors,
    )


# =========================================================
# Evaluator service
# =========================================================

class FailureEvaluator:

    def evaluate(
        self,
        samples: list[
            FailureEvaluationSample
        ],
        *,
        minimum_actionable_hours: float = 24.0,
    ) -> FailureEvaluationResult:

        return evaluate_failures(
            samples,
            minimum_actionable_hours=(
                minimum_actionable_hours
            ),
        )

    def threshold(
        self,
        samples: list[
            FailureEvaluationSample
        ],
        *,
        threshold: float,
    ) -> dict[str, Any]:

        return evaluate_failure_threshold(
            samples,
            threshold=threshold,
        )

    def best_threshold(
        self,
        samples: list[
            FailureEvaluationSample
        ],
        *,
        thresholds: Iterable[
            float
        ] | None = None,
    ) -> dict[str, Any]:

        return (
            find_best_failure_threshold(
                samples,
                thresholds=thresholds,
            )
        )

    def critical_recall(
        self,
        samples: list[
            FailureEvaluationSample
        ],
    ) -> float | None:

        return (
            calculate_critical_failure_recall(
                samples
            )
        )


failure_evaluator = FailureEvaluator()


__all__ = [
    "FailureType",
    "FailureSeverity",
    "FailurePrediction",
    "FailureGroundTruth",
    "FailureEvaluationSample",
    "BinaryFailureMetrics",
    "FailureClassMetrics",
    "FailureEvaluationResult",
    "calculate_binary_metrics",
    "build_failure_confusion_matrix",
    "calculate_failure_type_metrics",
    "calculate_failure_type_accuracy",
    "calculate_severity_accuracy",
    "calculate_critical_failure_recall",
    "calculate_brier_score",
    "calculate_ece",
    "get_actual_ttf_hours",
    "calculate_ttf_mae",
    "calculate_lead_time_metrics",
    "evaluate_failure_threshold",
    "find_best_failure_threshold",
    "evaluate_failures",
    "FailureEvaluator",
    "failure_evaluator",
]