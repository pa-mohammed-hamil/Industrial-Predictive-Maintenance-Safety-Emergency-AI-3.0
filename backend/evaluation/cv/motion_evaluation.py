# app/evaluation/motion_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from statistics import mean
from typing import Any


# =========================================================
# Motion classes
# =========================================================

class MotionClass(str, Enum):
    NO_MOTION = "no_motion"
    NORMAL_MOTION = "normal_motion"

    PERSON_MOTION = "person_motion"
    MACHINE_MOTION = "machine_motion"
    VEHICLE_MOTION = "vehicle_motion"

    RESTRICTED_ZONE_ENTRY = "restricted_zone_entry"
    UNAUTHORIZED_MOTION = "unauthorized_motion"

    FALL = "fall"
    RUNNING = "running"
    ABNORMAL_MOTION = "abnormal_motion"

    UNSAFE_PROXIMITY = "unsafe_proximity"

    OTHER = "other"


class MotionSeverity(str, Enum):
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
# Prediction / ground truth
# =========================================================

@dataclass(slots=True)
class MotionPrediction:
    motion_class: MotionClass

    confidence: float

    severity: MotionSeverity = (
        MotionSeverity.NONE
    )

    bounding_box: BoundingBox | None = None

    track_id: str | None = None

    speed_mps: float | None = None

    model_name: str | None = None
    model_version: str | None = None


@dataclass(slots=True)
class MotionGroundTruth:
    motion_class: MotionClass

    severity: MotionSeverity = (
        MotionSeverity.NONE
    )

    bounding_box: BoundingBox | None = None

    track_id: str | None = None

    speed_mps: float | None = None


# =========================================================
# Evaluation sample
# =========================================================

@dataclass(slots=True)
class MotionEvaluationSample:
    sample_id: str
    asset_id: str

    prediction: MotionPrediction
    ground_truth: MotionGroundTruth

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
class MotionClassMetrics:
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
# Overall result
# =========================================================

@dataclass(slots=True)
class MotionEvaluationResult:
    total_samples: int

    correct_predictions: int
    incorrect_predictions: int

    accuracy: float

    motion_detection_accuracy: float

    precision: float
    recall: float
    specificity: float
    f1_score: float

    false_positive_rate: float
    false_negative_rate: float

    missed_motion_events: int
    false_motion_alarms: int

    macro_precision: float
    macro_recall: float
    macro_f1: float

    critical_motion_recall: float | None

    restricted_zone_recall: float | None
    fall_detection_recall: float | None
    unsafe_proximity_recall: float | None

    severity_accuracy: float | None

    mean_iou: float | None
    localization_accuracy: float | None

    speed_mae_mps: float | None

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
        MotionClassMetrics,
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_class"] = {
            name: metrics.to_dict()
            for name, metrics
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

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


# =========================================================
# Motion helpers
# =========================================================

def _is_motion(
    motion_class: MotionClass,
) -> bool:

    return motion_class not in {
        MotionClass.NO_MOTION,
    }


def _is_safety_motion(
    motion_class: MotionClass,
) -> bool:

    return motion_class in {
        MotionClass.RESTRICTED_ZONE_ENTRY,
        MotionClass.UNAUTHORIZED_MOTION,
        MotionClass.FALL,
        MotionClass.ABNORMAL_MOTION,
        MotionClass.UNSAFE_PROXIMITY,
    }


# =========================================================
# IoU
# =========================================================

def calculate_iou(
    box_a: BoundingBox,
    box_b: BoundingBox,
) -> float:

    x1 = max(
        box_a.x1,
        box_b.x1,
    )

    y1 = max(
        box_a.y1,
        box_b.y1,
    )

    x2 = min(
        box_a.x2,
        box_b.x2,
    )

    y2 = min(
        box_a.y2,
        box_b.y2,
    )

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
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
        MotionEvaluationSample
    ],
) -> dict[str, dict[str, int]]:

    labels = [
        item.value
        for item in MotionClass
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
            .motion_class
            .value
        )

        predicted = (
            sample
            .prediction
            .motion_class
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
    label: MotionClass,
    matrix: dict[
        str,
        dict[str, int],
    ],
) -> MotionClassMetrics:

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

    return MotionClassMetrics(
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

        support=(
            tp + fn
        ),
    )


# =========================================================
# Binary motion evaluation
# =========================================================

def evaluate_binary_motion(
    samples: list[
        MotionEvaluationSample
    ],
) -> dict[str, Any]:

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in samples:
        actual = _is_motion(
            sample.ground_truth.motion_class
        )

        predicted = _is_motion(
            sample.prediction.motion_class
        )

        if actual and predicted:
            tp += 1

        elif not actual and not predicted:
            tn += 1

        elif not actual and predicted:
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
# Specific event recall
# =========================================================

def calculate_event_recall(
    samples: list[
        MotionEvaluationSample
    ],
    event: MotionClass,
) -> float | None:

    relevant = [
        sample
        for sample in samples
        if (
            sample.ground_truth.motion_class
            == event
        )
    ]

    if not relevant:
        return None

    detected = sum(
        sample.prediction.motion_class
        == event
        for sample in relevant
    )

    return round(
        detected / len(relevant),
        4,
    )


# =========================================================
# Critical safety recall
# =========================================================

def calculate_critical_motion_recall(
    samples: list[
        MotionEvaluationSample
    ],
) -> float | None:

    critical = [
        sample
        for sample in samples
        if (
            sample.ground_truth.severity
            == MotionSeverity.CRITICAL
            and _is_safety_motion(
                sample.ground_truth.motion_class
            )
        )
    ]

    if not critical:
        return None

    detected = sum(
        _is_safety_motion(
            sample.prediction.motion_class
        )
        for sample in critical
    )

    return round(
        detected / len(critical),
        4,
    )


# =========================================================
# Severity accuracy
# =========================================================

def evaluate_severity_accuracy(
    samples: list[
        MotionEvaluationSample
    ],
) -> float | None:

    relevant = [
        sample
        for sample in samples
        if _is_safety_motion(
            sample.ground_truth.motion_class
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
        MotionEvaluationSample
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
# Speed estimation
# =========================================================

def calculate_speed_mae(
    samples: list[
        MotionEvaluationSample
    ],
) -> float | None:

    errors: list[float] = []

    for sample in samples:
        predicted = (
            sample.prediction.speed_mps
        )

        actual = (
            sample.ground_truth.speed_mps
        )

        if (
            predicted is None
            or actual is None
        ):
            continue

        errors.append(
            abs(
                float(predicted)
                - float(actual)
            )
        )

    if not errors:
        return None

    return round(
        mean(errors),
        4,
    )


# =========================================================
# Confidence calibration
# =========================================================

def calculate_ece(
    samples: list[
        MotionEvaluationSample
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
            MotionEvaluationSample
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
                sample.prediction.motion_class
                == sample.ground_truth.motion_class
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
# Latency metrics
# =========================================================

def calculate_latency_metrics(
    samples: list[
        MotionEvaluationSample
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

    # Nearest-rank style p95.
    index = min(
        len(latencies) - 1,
        max(
            0,
            int(
                0.95
                * len(latencies)
            ),
        ),
    )

    return (
        average_latency,
        round(
            latencies[index],
            2,
        ),
    )


# =========================================================
# Main evaluator
# =========================================================

def evaluate_motion_detection(
    samples: list[
        MotionEvaluationSample
    ],
    *,
    iou_threshold: float = 0.5,
) -> MotionEvaluationResult:

    if not samples:
        return MotionEvaluationResult(
            total_samples=0,

            correct_predictions=0,
            incorrect_predictions=0,

            accuracy=0.0,
            motion_detection_accuracy=0.0,

            precision=0.0,
            recall=0.0,
            specificity=0.0,
            f1_score=0.0,

            false_positive_rate=0.0,
            false_negative_rate=0.0,

            missed_motion_events=0,
            false_motion_alarms=0,

            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,

            critical_motion_recall=None,

            restricted_zone_recall=None,
            fall_detection_recall=None,
            unsafe_proximity_recall=None,

            severity_accuracy=None,

            mean_iou=None,
            localization_accuracy=None,

            speed_mae_mps=None,

            average_confidence=0.0,
            expected_calibration_error=0.0,

            average_detection_latency_ms=None,
            p95_detection_latency_ms=None,

            confusion_matrix={},
            per_class={},
            errors=[],
        )

    for sample in samples:
        _validate_probability(
            sample.prediction.confidence,
            "confidence",
        )

    total = len(samples)

    correct = sum(
        sample.prediction.motion_class
        == sample.ground_truth.motion_class
        for sample in samples
    )

    accuracy = (
        correct / total
    )

    # -----------------------------------------------------
    # Confusion matrix
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
        for label in MotionClass
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
    # Binary motion
    # -----------------------------------------------------

    binary = evaluate_binary_motion(
        samples
    )

    # -----------------------------------------------------
    # Localization
    # -----------------------------------------------------

    localization = evaluate_localization(
        samples,
        iou_threshold=iou_threshold,
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
    # Errors
    # -----------------------------------------------------

    errors: list[
        dict[str, Any]
    ] = []

    for sample in samples:
        predicted = (
            sample.prediction.motion_class
        )

        actual = (
            sample.ground_truth.motion_class
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
            and
            sample.ground_truth.bounding_box
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
            or localization_failure
            or (
                _is_safety_motion(actual)
                and not severity_correct
            )
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

                    "actual_severity":
                        sample.ground_truth
                        .severity.value,

                    "predicted_severity":
                        sample.prediction
                        .severity.value,

                    "iou":
                        iou,

                    "predicted_speed_mps":
                        sample.prediction.speed_mps,

                    "actual_speed_mps":
                        sample.ground_truth.speed_mps,

                    "detection_latency_ms":
                        sample.detection_latency_ms,
                }
            )

    return MotionEvaluationResult(
        total_samples=total,

        correct_predictions=correct,

        incorrect_predictions=(
            total - correct
        ),

        accuracy=round(
            accuracy,
            4,
        ),

        motion_detection_accuracy=(
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

        missed_motion_events=(
            binary["fn"]
        ),

        false_motion_alarms=(
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

        critical_motion_recall=(
            calculate_critical_motion_recall(
                samples
            )
        ),

        restricted_zone_recall=(
            calculate_event_recall(
                samples,
                MotionClass.RESTRICTED_ZONE_ENTRY,
            )
        ),

        fall_detection_recall=(
            calculate_event_recall(
                samples,
                MotionClass.FALL,
            )
        ),

        unsafe_proximity_recall=(
            calculate_event_recall(
                samples,
                MotionClass.UNSAFE_PROXIMITY,
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

        speed_mae_mps=(
            calculate_speed_mae(
                samples
            )
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

def evaluate_motion_threshold(
    samples: list[
        MotionEvaluationSample
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
        correct = sum(
            sample.prediction.motion_class
            == sample.ground_truth.motion_class
            for sample in accepted
        )

        accuracy = (
            correct
            / len(accepted)
        )

        binary = (
            evaluate_binary_motion(
                accepted
            )
        )

    else:
        accuracy = 0.0

        binary = {
            "recall": 0.0,
            "false_positive_rate": 0.0,
        }

    coverage = (
        len(accepted)
        / len(samples)
        if samples
        else 0.0
    )

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
            coverage,
            4,
        ),

        "accepted_accuracy": round(
            accuracy,
            4,
        ),

        "motion_recall":
            binary["recall"],

        "false_alarm_rate":
            binary[
                "false_positive_rate"
            ],
    }


# =========================================================
# Evaluator
# =========================================================

class MotionEvaluator:

    def evaluate(
        self,
        samples: list[
            MotionEvaluationSample
        ],
        *,
        iou_threshold: float = 0.5,
    ) -> MotionEvaluationResult:

        return evaluate_motion_detection(
            samples,
            iou_threshold=iou_threshold,
        )

    def threshold(
        self,
        samples: list[
            MotionEvaluationSample
        ],
        *,
        threshold: float,
    ) -> dict[str, Any]:

        return evaluate_motion_threshold(
            samples,
            threshold=threshold,
        )

    def critical_recall(
        self,
        samples: list[
            MotionEvaluationSample
        ],
    ) -> float | None:

        return (
            calculate_critical_motion_recall(
                samples
            )
        )


motion_evaluator = MotionEvaluator()


__all__ = [
    "MotionClass",
    "MotionSeverity",
    "BoundingBox",
    "MotionPrediction",
    "MotionGroundTruth",
    "MotionEvaluationSample",
    "MotionClassMetrics",
    "MotionEvaluationResult",
    "calculate_iou",
    "build_confusion_matrix",
    "calculate_class_metrics",
    "evaluate_binary_motion",
    "calculate_event_recall",
    "calculate_critical_motion_recall",
    "evaluate_severity_accuracy",
    "evaluate_localization",
    "calculate_speed_mae",
    "calculate_ece",
    "calculate_latency_metrics",
    "evaluate_motion_detection",
    "evaluate_motion_threshold",
    "MotionEvaluator",
    "motion_evaluator",
]