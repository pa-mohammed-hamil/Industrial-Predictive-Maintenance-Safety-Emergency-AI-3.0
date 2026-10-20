# app/evaluation/defect_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from statistics import mean
from typing import Any, Iterable


# =========================================================
# Defect types
# =========================================================

class DefectType(str, Enum):
    NONE = "none"

    CRACK = "crack"
    CORROSION = "corrosion"
    WEAR = "wear"
    LEAKAGE = "leakage"

    DEFORMATION = "deformation"
    DENT = "dent"
    SCRATCH = "scratch"

    SURFACE_DAMAGE = "surface_damage"
    STRUCTURAL_DAMAGE = "structural_damage"

    OVERHEATING = "overheating"
    BURN_DAMAGE = "burn_damage"

    ELECTRICAL_DAMAGE = "electrical_damage"

    LOOSE_COMPONENT = "loose_component"
    MISSING_COMPONENT = "missing_component"

    CONTAMINATION = "contamination"

    OTHER = "other"


class DefectSeverity(str, Enum):
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
class DefectPrediction:
    defect_type: DefectType

    confidence: float

    bounding_box: (
        BoundingBox | None
    ) = None

    severity: DefectSeverity = (
        DefectSeverity.NONE
    )

    model_name: str | None = None
    model_version: str | None = None


# =========================================================
# Ground truth
# =========================================================

@dataclass(slots=True)
class DefectGroundTruth:
    defect_type: DefectType

    bounding_box: (
        BoundingBox | None
    ) = None

    severity: DefectSeverity = (
        DefectSeverity.NONE
    )


# =========================================================
# Evaluation sample
# =========================================================

@dataclass(slots=True)
class DefectEvaluationSample:
    sample_id: str
    asset_id: str

    prediction: DefectPrediction
    ground_truth: DefectGroundTruth

    image_id: str | None = None
    inspection_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Per-class metrics
# =========================================================

@dataclass(slots=True)
class DefectClassMetrics:
    defect_type: str

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


# =========================================================
# Overall result
# =========================================================

@dataclass(slots=True)
class DefectEvaluationResult:
    total_samples: int

    correct_classifications: int
    incorrect_classifications: int

    accuracy: float

    precision: float
    recall: float
    f1_score: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    defect_detection_accuracy: float

    false_positive_rate: float
    false_negative_rate: float

    missed_defects: int
    false_alarms: int

    severity_accuracy: float | None

    localization_accuracy: float | None
    mean_iou: float | None

    average_confidence: float

    expected_calibration_error: float

    per_class: dict[
        str,
        DefectClassMetrics,
    ]

    confusion_matrix: dict[
        str,
        dict[str, int],
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples":
                self.total_samples,

            "correct_classifications":
                self.correct_classifications,

            "incorrect_classifications":
                self.incorrect_classifications,

            "accuracy":
                self.accuracy,

            "precision":
                self.precision,

            "recall":
                self.recall,

            "f1_score":
                self.f1_score,

            "macro_precision":
                self.macro_precision,

            "macro_recall":
                self.macro_recall,

            "macro_f1":
                self.macro_f1,

            "defect_detection_accuracy":
                self.defect_detection_accuracy,

            "false_positive_rate":
                self.false_positive_rate,

            "false_negative_rate":
                self.false_negative_rate,

            "missed_defects":
                self.missed_defects,

            "false_alarms":
                self.false_alarms,

            "severity_accuracy":
                self.severity_accuracy,

            "localization_accuracy":
                self.localization_accuracy,

            "mean_iou":
                self.mean_iou,

            "average_confidence":
                self.average_confidence,

            "expected_calibration_error":
                self.expected_calibration_error,

            "per_class": {
                key: value.to_dict()
                for key, value
                in self.per_class.items()
            },

            "confusion_matrix":
                self.confusion_matrix,

            "errors":
                self.errors,
        }


# =========================================================
# Validation
# =========================================================

def _validate_confidence(
    value: float,
) -> float:

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "Confidence must be "
            "between 0 and 1."
        )

    return value


# =========================================================
# IoU
# =========================================================

def calculate_iou(
    box_a: BoundingBox,
    box_b: BoundingBox,
) -> float:
    """
    Intersection over Union.

                  intersection
        IoU = -------------------
               union of two boxes
    """

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
    samples: Iterable[
        DefectEvaluationSample
    ],
) -> dict[str, dict[str, int]]:

    labels = [
        item.value
        for item in DefectType
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
            .defect_type
            .value
        )

        predicted = (
            sample
            .prediction
            .defect_type
            .value
        )

        matrix[
            actual
        ][
            predicted
        ] += 1

    return matrix


# =========================================================
# Per-class evaluation
# =========================================================

def calculate_class_metrics(
    *,
    defect_type: DefectType,
    matrix: dict[
        str,
        dict[str, int],
    ],
) -> DefectClassMetrics:

    label = defect_type.value

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
        for predicted
        in matrix[label]
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
        if (
            precision
            + recall
        )
        else 0.0
    )

    return DefectClassMetrics(
        defect_type=label,

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
# Binary defect detection
# =========================================================

def evaluate_binary_detection(
    samples: list[
        DefectEvaluationSample
    ],
) -> dict[str, Any]:

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in samples:
        actual_defect = (
            sample.ground_truth.defect_type
            != DefectType.NONE
        )

        predicted_defect = (
            sample.prediction.defect_type
            != DefectType.NONE
        )

        if (
            actual_defect
            and predicted_defect
        ):
            tp += 1

        elif (
            not actual_defect
            and not predicted_defect
        ):
            tn += 1

        elif (
            not actual_defect
            and predicted_defect
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

    accuracy = (
        (tp + tn) / total
        if total
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
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,

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
# Localization evaluation
# =========================================================

def evaluate_localization(
    samples: list[
        DefectEvaluationSample
    ],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | None]:

    if not 0 <= iou_threshold <= 1:
        raise ValueError(
            "iou_threshold must be "
            "between 0 and 1."
        )

    ious: list[float] = []

    correct = 0

    for sample in samples:
        predicted_box = (
            sample.prediction
            .bounding_box
        )

        actual_box = (
            sample.ground_truth
            .bounding_box
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
# Severity evaluation
# =========================================================

def evaluate_severity(
    samples: list[
        DefectEvaluationSample
    ],
) -> float | None:

    relevant = [
        sample
        for sample in samples
        if (
            sample.ground_truth.defect_type
            != DefectType.NONE
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
# Confidence calibration
# =========================================================

def calculate_ece(
    samples: list[
        DefectEvaluationSample
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
            DefectEvaluationSample
        ]
    ] = [
        []
        for _ in range(bins)
    ]

    for sample in samples:
        confidence = (
            _validate_confidence(
                sample.prediction.confidence
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

    error = 0.0

    for bucket in buckets:
        if not bucket:
            continue

        accuracy = mean(
            1.0
            if (
                sample.prediction.defect_type
                == sample.ground_truth.defect_type
            )
            else 0.0
            for sample in bucket
        )

        confidence = mean(
            sample.prediction.confidence
            for sample in bucket
        )

        error += (
            len(bucket)
            / len(samples)
        ) * abs(
            accuracy
            - confidence
        )

    return round(
        error,
        4,
    )


# =========================================================
# Main evaluation
# =========================================================

def evaluate_defects(
    samples: list[
        DefectEvaluationSample
    ],
    *,
    iou_threshold: float = 0.5,
) -> DefectEvaluationResult:

    if not samples:
        return DefectEvaluationResult(
            total_samples=0,

            correct_classifications=0,
            incorrect_classifications=0,

            accuracy=0.0,

            precision=0.0,
            recall=0.0,
            f1_score=0.0,

            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,

            defect_detection_accuracy=0.0,

            false_positive_rate=0.0,
            false_negative_rate=0.0,

            missed_defects=0,
            false_alarms=0,

            severity_accuracy=None,

            localization_accuracy=None,
            mean_iou=None,

            average_confidence=0.0,

            expected_calibration_error=0.0,

            per_class={},
            confusion_matrix={},

            errors=[],
        )

    for sample in samples:
        _validate_confidence(
            sample.prediction.confidence
        )

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    correct = sum(
        sample.prediction.defect_type
        == sample.ground_truth.defect_type
        for sample in samples
    )

    total = len(samples)

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
        defect.value:
            calculate_class_metrics(
                defect_type=defect,
                matrix=matrix,
            )
        for defect in DefectType
    }

    supported = [
        metric
        for metric
        in per_class.values()
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
    # Binary defect detection
    # -----------------------------------------------------

    binary = (
        evaluate_binary_detection(
            samples
        )
    )

    # -----------------------------------------------------
    # Localization
    # -----------------------------------------------------

    localization = (
        evaluate_localization(
            samples,
            iou_threshold=(
                iou_threshold
            ),
        )
    )

    # -----------------------------------------------------
    # Errors
    # -----------------------------------------------------

    errors: list[
        dict[str, Any]
    ] = []

    for sample in samples:
        predicted = (
            sample.prediction.defect_type
        )

        actual = (
            sample.ground_truth.defect_type
        )

        severity_correct = (
            sample.prediction.severity
            == sample.ground_truth.severity
        )

        classification_correct = (
            predicted == actual
        )

        localization_iou = None

        if (
            sample.prediction.bounding_box
            is not None
            and
            sample.ground_truth.bounding_box
            is not None
        ):
            localization_iou = (
                calculate_iou(
                    sample.prediction
                    .bounding_box,

                    sample.ground_truth
                    .bounding_box,
                )
            )

        if (
            not classification_correct
            or not severity_correct
            or (
                localization_iou
                is not None
                and localization_iou
                < iou_threshold
            )
        ):
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "asset_id":
                        sample.asset_id,

                    "image_id":
                        sample.image_id,

                    "predicted":
                        predicted.value,

                    "actual":
                        actual.value,

                    "confidence":
                        sample.prediction.confidence,

                    "predicted_severity":
                        sample.prediction
                        .severity.value,

                    "actual_severity":
                        sample.ground_truth
                        .severity.value,

                    "iou":
                        localization_iou,

                    "classification_correct":
                        classification_correct,

                    "severity_correct":
                        severity_correct,
                }
            )

    return DefectEvaluationResult(
        total_samples=total,

        correct_classifications=correct,

        incorrect_classifications=(
            total - correct
        ),

        accuracy=round(
            accuracy,
            4,
        ),

        precision=(
            binary["precision"]
        ),

        recall=(
            binary["recall"]
        ),

        f1_score=(
            binary["f1"]
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

        defect_detection_accuracy=(
            binary["accuracy"]
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

        missed_defects=(
            binary["false_negative"]
        ),

        false_alarms=(
            binary["false_positive"]
        ),

        severity_accuracy=(
            evaluate_severity(
                samples
            )
        ),

        localization_accuracy=(
            localization[
                "localization_accuracy"
            ]
        ),

        mean_iou=(
            localization[
                "mean_iou"
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

        per_class=per_class,

        confusion_matrix=matrix,

        errors=errors,
    )


# =========================================================
# Confidence threshold evaluation
# =========================================================

def evaluate_confidence_threshold(
    samples: list[
        DefectEvaluationSample
    ],
    *,
    threshold: float,
) -> dict[str, Any]:

    _validate_confidence(
        threshold
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
            sample.prediction.defect_type
            == sample.ground_truth.defect_type
            for sample in accepted
        )

        accuracy = (
            correct
            / len(accepted)
        )

    else:
        accuracy = 0.0

    coverage = (
        len(accepted)
        / len(samples)
        if samples
        else 0.0
    )

    return {
        "threshold": threshold,

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
    }


# =========================================================
# Critical defect recall
# =========================================================

def calculate_critical_defect_recall(
    samples: list[
        DefectEvaluationSample
    ],
) -> float | None:
    """
    Safety-critical metric.

    Measures how often actual CRITICAL defects
    were successfully detected.
    """

    critical = [
        sample
        for sample in samples
        if (
            sample.ground_truth.severity
            == DefectSeverity.CRITICAL
        )
    ]

    if not critical:
        return None

    detected = sum(
        (
            sample.prediction.defect_type
            != DefectType.NONE
        )
        for sample in critical
    )

    return round(
        detected / len(critical),
        4,
    )


# =========================================================
# Evaluator service
# =========================================================

class DefectEvaluator:

    def evaluate(
        self,
        samples: list[
            DefectEvaluationSample
        ],
        *,
        iou_threshold: float = 0.5,
    ) -> DefectEvaluationResult:

        return evaluate_defects(
            samples,
            iou_threshold=(
                iou_threshold
            ),
        )

    def evaluate_threshold(
        self,
        samples: list[
            DefectEvaluationSample
        ],
        *,
        threshold: float,
    ) -> dict[str, Any]:

        return (
            evaluate_confidence_threshold(
                samples,
                threshold=threshold,
            )
        )

    def critical_recall(
        self,
        samples: list[
            DefectEvaluationSample
        ],
    ) -> float | None:

        return (
            calculate_critical_defect_recall(
                samples
            )
        )


defect_evaluator = DefectEvaluator()