# app/evaluation/fire_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from statistics import mean
from typing import Any


class FireClass(str, Enum):
    NO_FIRE = "no_fire"
    FIRE = "fire"
    SMOKE = "smoke"
    FIRE_AND_SMOKE = "fire_and_smoke"


class FireSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True, frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(
            0.0, self.y2 - self.y1
        )


@dataclass(slots=True)
class FirePrediction:
    fire_class: FireClass
    confidence: float

    severity: FireSeverity = FireSeverity.NONE
    bounding_box: BoundingBox | None = None

    fire_probability: float | None = None
    smoke_probability: float | None = None

    model_name: str | None = None
    model_version: str | None = None


@dataclass(slots=True)
class FireGroundTruth:
    fire_class: FireClass

    severity: FireSeverity = FireSeverity.NONE
    bounding_box: BoundingBox | None = None


@dataclass(slots=True)
class FireEvaluationSample:
    sample_id: str
    asset_id: str

    prediction: FirePrediction
    ground_truth: FireGroundTruth

    camera_id: str | None = None
    incident_id: str | None = None

    detection_latency_ms: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class FireClassMetrics:
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


@dataclass(slots=True)
class FireEvaluationResult:
    total_samples: int

    accuracy: float
    precision: float
    recall: float
    f1_score: float

    specificity: float

    false_positive_rate: float
    false_negative_rate: float

    missed_fire_events: int
    false_fire_alarms: int

    critical_fire_recall: float | None

    severity_accuracy: float | None

    mean_iou: float | None
    localization_accuracy: float | None

    average_confidence: float
    expected_calibration_error: float

    average_detection_latency_ms: float | None
    p95_detection_latency_ms: float | None

    macro_precision: float
    macro_recall: float
    macro_f1: float

    confusion_matrix: dict[
        str,
        dict[str, int],
    ]

    per_class: dict[
        str,
        FireClassMetrics,
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_class"] = {
            name: metrics.to_dict()
            for name, metrics in self.per_class.items()
        }

        return result


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


def calculate_iou(
    box_a: BoundingBox,
    box_b: BoundingBox,
) -> float:
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)

    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    intersection = width * height

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


def build_confusion_matrix(
    samples: list[FireEvaluationSample],
) -> dict[str, dict[str, int]]:
    labels = [
        item.value
        for item in FireClass
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
            sample.ground_truth.fire_class.value
        )

        predicted = (
            sample.prediction.fire_class.value
        )

        matrix[actual][predicted] += 1

    return matrix


def calculate_class_metrics(
    *,
    label: FireClass,
    matrix: dict[str, dict[str, int]],
) -> FireClassMetrics:
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

    tn = total - tp - fp - fn

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

    return FireClassMetrics(
        label=name,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        specificity=round(
            specificity,
            4,
        ),
        f1_score=round(f1, 4),
        support=tp + fn,
    )


def _is_fire_event(
    fire_class: FireClass,
) -> bool:
    return fire_class in {
        FireClass.FIRE,
        FireClass.FIRE_AND_SMOKE,
    }


def evaluate_binary_fire_detection(
    samples: list[FireEvaluationSample],
) -> dict[str, Any]:
    tp = tn = fp = fn = 0

    for sample in samples:
        actual = _is_fire_event(
            sample.ground_truth.fire_class
        )

        predicted = _is_fire_event(
            sample.prediction.fire_class
        )

        if actual and predicted:
            tp += 1

        elif not actual and not predicted:
            tn += 1

        elif not actual and predicted:
            fp += 1

        else:
            fn += 1

    total = tp + tn + fp + fn

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

    accuracy = (
        (tp + tn) / total
        if total
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
            fpr,
            4,
        ),

        "false_negative_rate": round(
            fnr,
            4,
        ),
    }


def evaluate_severity_accuracy(
    samples: list[FireEvaluationSample],
) -> float | None:
    relevant = [
        sample
        for sample in samples
        if _is_fire_event(
            sample.ground_truth.fire_class
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


def evaluate_critical_fire_recall(
    samples: list[FireEvaluationSample],
) -> float | None:
    critical = [
        sample
        for sample in samples
        if (
            sample.ground_truth.severity
            == FireSeverity.CRITICAL
            and _is_fire_event(
                sample.ground_truth.fire_class
            )
        )
    ]

    if not critical:
        return None

    detected = sum(
        _is_fire_event(
            sample.prediction.fire_class
        )
        for sample in critical
    )

    return round(
        detected / len(critical),
        4,
    )


def evaluate_localization(
    samples: list[FireEvaluationSample],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | None]:
    _validate_probability(
        iou_threshold,
        "iou_threshold",
    )

    ious: list[float] = []

    localized = 0

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
            localized += 1

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
            localized / len(ious),
            4,
        ),
    }


def calculate_ece(
    samples: list[FireEvaluationSample],
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
        list[FireEvaluationSample]
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
                sample.prediction.fire_class
                == sample.ground_truth.fire_class
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

    return round(ece, 4)


def calculate_latency_metrics(
    samples: list[FireEvaluationSample],
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

    p95 = round(
        latencies[index],
        2,
    )

    return (
        average_latency,
        p95,
    )


def evaluate_fire_detection(
    samples: list[FireEvaluationSample],
    *,
    iou_threshold: float = 0.5,
) -> FireEvaluationResult:
    if not samples:
        return FireEvaluationResult(
            total_samples=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            specificity=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            missed_fire_events=0,
            false_fire_alarms=0,
            critical_fire_recall=None,
            severity_accuracy=None,
            mean_iou=None,
            localization_accuracy=None,
            average_confidence=0.0,
            expected_calibration_error=0.0,
            average_detection_latency_ms=None,
            p95_detection_latency_ms=None,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            confusion_matrix={},
            per_class={},
            errors=[],
        )

    for sample in samples:
        _validate_probability(
            sample.prediction.confidence,
            "confidence",
        )

    matrix = build_confusion_matrix(
        samples
    )

    per_class = {
        label.value:
            calculate_class_metrics(
                label=label,
                matrix=matrix,
            )
        for label in FireClass
    }

    supported = [
        metrics
        for metrics in per_class.values()
        if metrics.support > 0
    ]

    if supported:
        macro_precision = mean(
            item.precision
            for item in supported
        )

        macro_recall = mean(
            item.recall
            for item in supported
        )

        macro_f1 = mean(
            item.f1_score
            for item in supported
        )
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    binary = (
        evaluate_binary_fire_detection(
            samples
        )
    )

    localization = evaluate_localization(
        samples,
        iou_threshold=iou_threshold,
    )

    (
        average_latency,
        p95_latency,
    ) = calculate_latency_metrics(
        samples
    )

    errors: list[dict[str, Any]] = []

    for sample in samples:
        predicted = (
            sample.prediction.fire_class
        )

        actual = (
            sample.ground_truth.fire_class
        )

        classification_correct = (
            predicted == actual
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

        severity_correct = (
            sample.prediction.severity
            == sample.ground_truth.severity
        )

        if (
            not classification_correct
            or localization_failure
            or not severity_correct
        ):
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "asset_id":
                        sample.asset_id,

                    "camera_id":
                        sample.camera_id,

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

                    "iou": iou,

                    "detection_latency_ms":
                        sample.detection_latency_ms,

                    "classification_correct":
                        classification_correct,

                    "severity_correct":
                        severity_correct,
                }
            )

    return FireEvaluationResult(
        total_samples=len(samples),

        accuracy=binary[
            "accuracy"
        ],

        precision=binary[
            "precision"
        ],

        recall=binary[
            "recall"
        ],

        f1_score=binary[
            "f1"
        ],

        specificity=binary[
            "specificity"
        ],

        false_positive_rate=binary[
            "false_positive_rate"
        ],

        false_negative_rate=binary[
            "false_negative_rate"
        ],

        missed_fire_events=binary[
            "fn"
        ],

        false_fire_alarms=binary[
            "fp"
        ],

        critical_fire_recall=(
            evaluate_critical_fire_recall(
                samples
            )
        ),

        severity_accuracy=(
            evaluate_severity_accuracy(
                samples
            )
        ),

        mean_iou=localization[
            "mean_iou"
        ],

        localization_accuracy=localization[
            "localization_accuracy"
        ],

        average_confidence=round(
            mean(
                sample.prediction.confidence
                for sample in samples
            ),
            4,
        ),

        expected_calibration_error=(
            calculate_ece(samples)
        ),

        average_detection_latency_ms=(
            average_latency
        ),

        p95_detection_latency_ms=(
            p95_latency
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

        confusion_matrix=matrix,
        per_class=per_class,
        errors=errors,
    )


def evaluate_fire_threshold(
    samples: list[FireEvaluationSample],
    *,
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate automatic fire alerts at a chosen
    confidence threshold.
    """

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

    if not accepted:
        return {
            "threshold": threshold,
            "total": len(samples),
            "accepted": 0,
            "coverage": 0.0,
            "fire_recall": 0.0,
            "false_alarm_rate": 0.0,
        }

    metrics = (
        evaluate_binary_fire_detection(
            accepted
        )
    )

    return {
        "threshold": threshold,

        "total": len(samples),

        "accepted":
            len(accepted),

        "coverage": round(
            len(accepted)
            / len(samples),
            4,
        ) if samples else 0.0,

        "fire_recall":
            metrics["recall"],

        "false_alarm_rate":
            metrics["false_positive_rate"],
    }


class FireEvaluator:
    def evaluate(
        self,
        samples: list[
            FireEvaluationSample
        ],
        *,
        iou_threshold: float = 0.5,
    ) -> FireEvaluationResult:
        return evaluate_fire_detection(
            samples,
            iou_threshold=iou_threshold,
        )

    def threshold(
        self,
        samples: list[
            FireEvaluationSample
        ],
        *,
        threshold: float,
    ) -> dict[str, Any]:
        return evaluate_fire_threshold(
            samples,
            threshold=threshold,
        )

    def critical_recall(
        self,
        samples: list[
            FireEvaluationSample
        ],
    ) -> float | None:
        return (
            evaluate_critical_fire_recall(
                samples
            )
        )


fire_evaluator = FireEvaluator()


__all__ = [
    "FireClass",
    "FireSeverity",
    "BoundingBox",
    "FirePrediction",
    "FireGroundTruth",
    "FireEvaluationSample",
    "FireClassMetrics",
    "FireEvaluationResult",
    "calculate_iou",
    "build_confusion_matrix",
    "evaluate_binary_fire_detection",
    "evaluate_severity_accuracy",
    "evaluate_critical_fire_recall",
    "evaluate_localization",
    "calculate_ece",
    "calculate_latency_metrics",
    "evaluate_fire_detection",
    "evaluate_fire_threshold",
    "FireEvaluator",
    "fire_evaluator",
]