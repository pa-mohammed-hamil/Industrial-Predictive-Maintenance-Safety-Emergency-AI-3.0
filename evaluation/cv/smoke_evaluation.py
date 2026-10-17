# app/evaluation/smoke_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import ceil, isfinite
from statistics import mean
from typing import Any


# =========================================================
# Smoke classes
# =========================================================

class SmokeClass(str, Enum):
    NO_SMOKE = "no_smoke"

    LIGHT_SMOKE = "light_smoke"
    MODERATE_SMOKE = "moderate_smoke"
    DENSE_SMOKE = "dense_smoke"

    WHITE_SMOKE = "white_smoke"
    GREY_SMOKE = "grey_smoke"
    BLACK_SMOKE = "black_smoke"

    STEAM = "steam"
    DUST = "dust"

    UNKNOWN_SMOKE = "unknown_smoke"


class SmokeSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =========================================================
# Bounding box
# =========================================================

@dataclass(slots=True, frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.x2 - self.x1,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.y2 - self.y1,
        )

    @property
    def area(self) -> float:
        return (
            self.width
            * self.height
        )


# =========================================================
# Prediction
# =========================================================

@dataclass(slots=True)
class SmokePrediction:
    smoke_class: SmokeClass

    confidence: float

    severity: SmokeSeverity = (
        SmokeSeverity.NONE
    )

    bounding_box: BoundingBox | None = None

    smoke_probability: float | None = None

    model_name: str | None = None
    model_version: str | None = None


# =========================================================
# Ground truth
# =========================================================

@dataclass(slots=True)
class SmokeGroundTruth:
    smoke_class: SmokeClass

    severity: SmokeSeverity = (
        SmokeSeverity.NONE
    )

    bounding_box: BoundingBox | None = None


# =========================================================
# Evaluation sample
# =========================================================

@dataclass(slots=True)
class SmokeEvaluationSample:
    sample_id: str
    asset_id: str

    prediction: SmokePrediction
    ground_truth: SmokeGroundTruth

    camera_id: str | None = None
    zone_id: str | None = None
    incident_id: str | None = None

    detection_latency_ms: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Class metrics
# =========================================================

@dataclass(slots=True)
class SmokeClassMetrics:
    label: str

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
# Evaluation result
# =========================================================

@dataclass(slots=True)
class SmokeEvaluationResult:
    total_samples: int

    correct_predictions: int
    incorrect_predictions: int

    classification_accuracy: float

    smoke_detection_accuracy: float

    precision: float
    recall: float
    specificity: float
    f1_score: float

    false_positive_rate: float
    false_negative_rate: float

    missed_smoke_events: int
    false_smoke_alarms: int

    macro_precision: float
    macro_recall: float
    macro_f1: float

    critical_smoke_recall: float | None
    dense_smoke_recall: float | None
    black_smoke_recall: float | None

    severity_accuracy: float | None

    mean_iou: float | None
    localization_accuracy: float | None

    average_confidence: float
    expected_calibration_error: float

    average_detection_latency_ms: float | None
    p95_detection_latency_ms: float | None

    confusion_matrix: dict[
        str,
        dict[str, int],
    ]

    per_class: dict[
        str,
        SmokeClassMetrics,
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_class"] = {
            label: metrics.to_dict()
            for label, metrics
            in self.per_class.items()
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


# =========================================================
# Smoke helpers
# =========================================================

def is_smoke(
    smoke_class: SmokeClass,
) -> bool:
    """
    STEAM and DUST are treated as non-smoke
    negative classes so the evaluator measures
    common industrial false-alarm sources.
    """

    return smoke_class not in {
        SmokeClass.NO_SMOKE,
        SmokeClass.STEAM,
        SmokeClass.DUST,
    }


def is_high_risk_smoke(
    smoke_class: SmokeClass,
) -> bool:

    return smoke_class in {
        SmokeClass.DENSE_SMOKE,
        SmokeClass.BLACK_SMOKE,
    }


# =========================================================
# IoU
# =========================================================

def calculate_iou(
    box_a: BoundingBox,
    box_b: BoundingBox,
) -> float:

    intersection_x1 = max(
        box_a.x1,
        box_b.x1,
    )

    intersection_y1 = max(
        box_a.y1,
        box_b.y1,
    )

    intersection_x2 = min(
        box_a.x2,
        box_b.x2,
    )

    intersection_y2 = min(
        box_a.y2,
        box_b.y2,
    )

    width = max(
        0.0,
        intersection_x2
        - intersection_x1,
    )

    height = max(
        0.0,
        intersection_y2
        - intersection_y1,
    )

    intersection = (
        width * height
    )

    union = (
        box_a.area
        + box_b.area
        - intersection
    )

    if union <= 0:
        return 0.0

    return round(
        intersection / union,
        6,
    )


# =========================================================
# Confusion matrix
# =========================================================

def build_confusion_matrix(
    samples: list[
        SmokeEvaluationSample
    ],
) -> dict[str, dict[str, int]]:

    labels = [
        smoke_class.value
        for smoke_class in SmokeClass
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
            sample
            .ground_truth
            .smoke_class
            .value
        )

        predicted = (
            sample
            .prediction
            .smoke_class
            .value
        )

        matrix[
            actual
        ][
            predicted
        ] += 1

    return matrix


# =========================================================
# Per-class metrics
# =========================================================

def calculate_class_metrics(
    *,
    label: SmokeClass,
    matrix: dict[
        str,
        dict[str, int],
    ],
) -> SmokeClassMetrics:

    name = label.value

    total = sum(
        sum(row.values())
        for row in matrix.values()
    )

    tp = matrix[name][name]

    fp = sum(
        matrix[actual][name]
        for actual in matrix
        if actual != name
    )

    fn = sum(
        matrix[name][predicted]
        for predicted in matrix[name]
        if predicted != name
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

    return SmokeClassMetrics(
        label=name,

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

        support=tp + fn,
    )


# =========================================================
# Binary smoke detection
# =========================================================

def evaluate_binary_smoke_detection(
    samples: list[
        SmokeEvaluationSample
    ],
) -> dict[str, Any]:

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in samples:
        actual = is_smoke(
            sample.ground_truth.smoke_class
        )

        predicted = is_smoke(
            sample.prediction.smoke_class
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

    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if fn + tp
        else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "accuracy": round(
            accuracy,
            4,
        ),

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

        "f1": round(
            f1,
            4,
        ),

        "false_positive_rate": round(
            false_positive_rate,
            4,
        ),

        "false_negative_rate": round(
            false_negative_rate,
            4,
        ),
    }


# =========================================================
# Event-specific recall
# =========================================================

def calculate_class_recall(
    samples: list[
        SmokeEvaluationSample
    ],
    target: SmokeClass,
) -> float | None:

    relevant = [
        sample
        for sample in samples
        if (
            sample.ground_truth.smoke_class
            == target
        )
    ]

    if not relevant:
        return None

    correct = sum(
        sample.prediction.smoke_class
        == target
        for sample in relevant
    )

    return round(
        correct / len(relevant),
        4,
    )


# =========================================================
# Critical smoke recall
# =========================================================

def calculate_critical_smoke_recall(
    samples: list[
        SmokeEvaluationSample
    ],
) -> float | None:

    critical = [
        sample
        for sample in samples
        if (
            sample.ground_truth.severity
            == SmokeSeverity.CRITICAL
            and is_smoke(
                sample.ground_truth.smoke_class
            )
        )
    ]

    if not critical:
        return None

    detected = sum(
        is_smoke(
            sample.prediction.smoke_class
        )
        for sample in critical
    )

    return round(
        detected / len(critical),
        4,
    )


# =========================================================
# Severity evaluation
# =========================================================

def evaluate_severity_accuracy(
    samples: list[
        SmokeEvaluationSample
    ],
) -> float | None:

    relevant = [
        sample
        for sample in samples
        if is_smoke(
            sample.ground_truth.smoke_class
        )
    ]

    if not relevant:
        return None

    correct = sum(
        sample.prediction.severity
        == sample.ground_truth.severity
        for sample in relevant
    )

    return round(
        correct / len(relevant),
        4,
    )


# =========================================================
# Localization
# =========================================================

def evaluate_localization(
    samples: list[
        SmokeEvaluationSample
    ],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | None]:

    _validate_probability(
        iou_threshold,
        "iou_threshold",
    )

    ious: list[float] = []

    correct = 0

    for sample in samples:
        predicted_box = (
            sample.prediction.bounding_box
        )

        actual_box = (
            sample.ground_truth.bounding_box
        )

        if (
            predicted_box is None
            or actual_box is None
        ):
            continue

        iou = calculate_iou(
            predicted_box,
            actual_box,
        )

        ious.append(iou)

        if iou >= iou_threshold:
            correct += 1

    if not ious:
        return {
            "mean_iou": None,
            "localization_accuracy": None,
        }

    return {
        "mean_iou": round(
            mean(ious),
            4,
        ),

        "localization_accuracy": round(
            correct / len(ious),
            4,
        ),
    }


# =========================================================
# Confidence calibration
# =========================================================

def calculate_ece(
    samples: list[
        SmokeEvaluationSample
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
            SmokeEvaluationSample
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
            int(confidence * bins),
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
                sample.prediction.smoke_class
                == sample.ground_truth.smoke_class
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
# Latency
# =========================================================

def calculate_latency_metrics(
    samples: list[
        SmokeEvaluationSample
    ],
) -> tuple[
    float | None,
    float | None,
]:

    latencies = sorted(
        max(
            0.0,
            float(
                sample.detection_latency_ms
            ),
        )
        for sample in samples
        if (
            sample.detection_latency_ms
            is not None
        )
    )

    if not latencies:
        return None, None

    average_latency = round(
        mean(latencies),
        2,
    )

    # Nearest-rank P95.
    rank = max(
        1,
        ceil(
            0.95
            * len(latencies)
        ),
    )

    index = min(
        rank - 1,
        len(latencies) - 1,
    )

    p95 = round(
        latencies[index],
        2,
    )

    return (
        average_latency,
        p95,
    )


# =========================================================
# Main evaluation
# =========================================================

def evaluate_smoke_detection(
    samples: list[
        SmokeEvaluationSample
    ],
    *,
    iou_threshold: float = 0.5,
) -> SmokeEvaluationResult:

    if not samples:
        return SmokeEvaluationResult(
            total_samples=0,

            correct_predictions=0,
            incorrect_predictions=0,

            classification_accuracy=0.0,
            smoke_detection_accuracy=0.0,

            precision=0.0,
            recall=0.0,
            specificity=0.0,
            f1_score=0.0,

            false_positive_rate=0.0,
            false_negative_rate=0.0,

            missed_smoke_events=0,
            false_smoke_alarms=0,

            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,

            critical_smoke_recall=None,
            dense_smoke_recall=None,
            black_smoke_recall=None,

            severity_accuracy=None,

            mean_iou=None,
            localization_accuracy=None,

            average_confidence=0.0,
            expected_calibration_error=0.0,

            average_detection_latency_ms=None,
            p95_detection_latency_ms=None,

            confusion_matrix={},
            per_class={},
            errors=[],
        )

    _validate_probability(
        iou_threshold,
        "iou_threshold",
    )

    for sample in samples:
        _validate_probability(
            sample.prediction.confidence,
            "confidence",
        )

        if (
            sample.prediction.smoke_probability
            is not None
        ):
            _validate_probability(
                sample.prediction.smoke_probability,
                "smoke_probability",
            )

    total = len(samples)

    correct = sum(
        sample.prediction.smoke_class
        == sample.ground_truth.smoke_class
        for sample in samples
    )

    classification_accuracy = (
        correct / total
    )

    # -----------------------------------------------------
    # Confusion matrix / multiclass metrics
    # -----------------------------------------------------

    matrix = build_confusion_matrix(
        samples
    )

    per_class = {
        label.value:
            calculate_class_metrics(
                label=label,
                matrix=matrix,
            )
        for label in SmokeClass
    }

    supported = [
        metric
        for metric in per_class.values()
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

    # -----------------------------------------------------
    # Binary smoke metrics
    # -----------------------------------------------------

    binary = (
        evaluate_binary_smoke_detection(
            samples
        )
    )

    # -----------------------------------------------------
    # Localization
    # -----------------------------------------------------

    localization = (
        evaluate_localization(
            samples,
            iou_threshold=iou_threshold,
        )
    )

    # -----------------------------------------------------
    # Latency
    # -----------------------------------------------------

    (
        average_latency,
        p95_latency,
    ) = calculate_latency_metrics(
        samples
    )

    # -----------------------------------------------------
    # Error analysis
    # -----------------------------------------------------

    errors: list[
        dict[str, Any]
    ] = []

    for sample in samples:
        predicted = (
            sample.prediction.smoke_class
        )

        actual = (
            sample.ground_truth.smoke_class
        )

        classification_correct = (
            predicted == actual
        )

        severity_correct = (
            sample.prediction.severity
            == sample.ground_truth.severity
        )

        iou: float | None = None

        if (
            sample.prediction.bounding_box
            is not None
            and sample.ground_truth.bounding_box
            is not None
        ):
            iou = calculate_iou(
                sample.prediction.bounding_box,
                sample.ground_truth.bounding_box,
            )

        localization_failure = (
            iou is not None
            and iou < iou_threshold
        )

        if (
            not classification_correct
            or (
                is_smoke(actual)
                and not severity_correct
            )
            or localization_failure
        ):
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "asset_id":
                        sample.asset_id,

                    "camera_id":
                        sample.camera_id,

                    "zone_id":
                        sample.zone_id,

                    "incident_id":
                        sample.incident_id,

                    "actual":
                        actual.value,

                    "predicted":
                        predicted.value,

                    "confidence":
                        sample.prediction.confidence,

                    "smoke_probability":
                        sample.prediction
                        .smoke_probability,

                    "actual_severity":
                        sample.ground_truth
                        .severity.value,

                    "predicted_severity":
                        sample.prediction
                        .severity.value,

                    "iou":
                        iou,

                    "detection_latency_ms":
                        sample.detection_latency_ms,

                    "classification_correct":
                        classification_correct,

                    "severity_correct":
                        severity_correct,
                }
            )

    return SmokeEvaluationResult(
        total_samples=total,

        correct_predictions=correct,

        incorrect_predictions=(
            total - correct
        ),

        classification_accuracy=round(
            classification_accuracy,
            4,
        ),

        smoke_detection_accuracy=(
            binary["accuracy"]
        ),

        precision=(
            binary["precision"]
        ),

        recall=(
            binary["recall"]
        ),

        specificity=(
            binary["specificity"]
        ),

        f1_score=(
            binary["f1"]
        ),

        false_positive_rate=(
            binary[
                "false_positive_rate"
            ]
        ),

        false_negative_rate=(
            binary[
                "false_negative_rate"
            ]
        ),

        missed_smoke_events=(
            binary["fn"]
        ),

        false_smoke_alarms=(
            binary["fp"]
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

        critical_smoke_recall=(
            calculate_critical_smoke_recall(
                samples
            )
        ),

        dense_smoke_recall=(
            calculate_class_recall(
                samples,
                SmokeClass.DENSE_SMOKE,
            )
        ),

        black_smoke_recall=(
            calculate_class_recall(
                samples,
                SmokeClass.BLACK_SMOKE,
            )
        ),

        severity_accuracy=(
            evaluate_severity_accuracy(
                samples
            )
        ),

        mean_iou=(
            localization["mean_iou"]
        ),

        localization_accuracy=(
            localization[
                "localization_accuracy"
            ]
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

        average_detection_latency_ms=(
            average_latency
        ),

        p95_detection_latency_ms=(
            p95_latency
        ),

        confusion_matrix=matrix,
        per_class=per_class,
        errors=errors,
    )


# =========================================================
# Confidence threshold evaluation
# =========================================================

def evaluate_smoke_threshold(
    samples: list[
        SmokeEvaluationSample
    ],
    *,
    threshold: float,
) -> dict[str, Any]:

    _validate_probability(
        threshold,
        "threshold",
    )

    accepted = [
        sample
        for sample in samples
        if (
            sample.prediction.confidence
            >= threshold
        )
    ]

    rejected = [
        sample
        for sample in samples
        if (
            sample.prediction.confidence
            < threshold
        )
    ]

    if accepted:
        metrics = (
            evaluate_binary_smoke_detection(
                accepted
            )
        )

        class_correct = sum(
            sample.prediction.smoke_class
            == sample.ground_truth.smoke_class
            for sample in accepted
        )

        class_accuracy = (
            class_correct
            / len(accepted)
        )

    else:
        metrics = {
            "recall": 0.0,
            "precision": 0.0,
            "false_positive_rate": 0.0,
        }

        class_accuracy = 0.0

    return {
        "threshold":
            threshold,

        "total_samples":
            len(samples),

        "accepted":
            len(accepted),

        "rejected":
            len(rejected),

        "coverage": round(
            (
                len(accepted)
                / len(samples)
            )
            if samples
            else 0.0,
            4,
        ),

        "accepted_accuracy": round(
            class_accuracy,
            4,
        ),

        "smoke_precision":
            metrics["precision"],

        "smoke_recall":
            metrics["recall"],

        "false_alarm_rate":
            metrics[
                "false_positive_rate"
            ],
    }


# =========================================================
# Evaluator service
# =========================================================

class SmokeEvaluator:

    def evaluate(
        self,
        samples: list[
            SmokeEvaluationSample
        ],
        *,
        iou_threshold: float = 0.5,
    ) -> SmokeEvaluationResult:

        return evaluate_smoke_detection(
            samples,
            iou_threshold=iou_threshold,
        )

    def threshold(
        self,
        samples: list[
            SmokeEvaluationSample
        ],
        *,
        threshold: float,
    ) -> dict[str, Any]:

        return evaluate_smoke_threshold(
            samples,
            threshold=threshold,
        )

    def critical_recall(
        self,
        samples: list[
            SmokeEvaluationSample
        ],
    ) -> float | None:

        return (
            calculate_critical_smoke_recall(
                samples
            )
        )


smoke_evaluator = SmokeEvaluator()


__all__ = [
    "SmokeClass",
    "SmokeSeverity",
    "BoundingBox",
    "SmokePrediction",
    "SmokeGroundTruth",
    "SmokeEvaluationSample",
    "SmokeClassMetrics",
    "SmokeEvaluationResult",
    "is_smoke",
    "is_high_risk_smoke",
    "calculate_iou",
    "build_confusion_matrix",
    "calculate_class_metrics",
    "evaluate_binary_smoke_detection",
    "calculate_class_recall",
    "calculate_critical_smoke_recall",
    "evaluate_severity_accuracy",
    "evaluate_localization",
    "calculate_ece",
    "calculate_latency_metrics",
    "evaluate_smoke_detection",
    "evaluate_smoke_threshold",
    "SmokeEvaluator",
    "smoke_evaluator",
]