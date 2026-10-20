# app/evaluation/repair_replacement_accuracy.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from statistics import mean
from typing import Any, Iterable


# =========================================================
# Decision classes
# =========================================================

class RepairReplacementDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    MONITOR = "monitor"
    INSPECT = "inspect"
    NO_ACTION = "no_action"


DECISION_LABELS = tuple(
    decision.value
    for decision in RepairReplacementDecision
)


# =========================================================
# Evaluation record
# =========================================================

@dataclass(slots=True)
class DecisionEvaluationRecord:
    asset_id: str

    predicted_decision: RepairReplacementDecision
    actual_decision: RepairReplacementDecision

    confidence: float = 1.0

    # Estimated cost of AI recommendation.
    predicted_action_cost: float = 0.0

    # Cost of the optimal/actual action.
    optimal_action_cost: float = 0.0

    # Realized financial outcome after following action.
    realized_total_cost: float | None = None

    # Estimated failure consequence if wrong action chosen.
    failure_consequence_cost: float = 0.0

    # Optional IDs for traceability.
    prediction_id: str | None = None
    recommendation_id: str | None = None
    work_order_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Class metrics
# =========================================================

@dataclass(slots=True)
class ClassMetrics:
    label: str

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
# Evaluation result
# =========================================================

@dataclass(slots=True)
class RepairReplacementAccuracyResult:
    total_samples: int

    correct_predictions: int
    incorrect_predictions: int

    accuracy: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    weighted_precision: float
    weighted_recall: float
    weighted_f1: float

    repair_accuracy: float | None
    replacement_accuracy: float | None

    repair_vs_replace_accuracy: float | None

    average_confidence: float
    correct_average_confidence: float | None
    incorrect_average_confidence: float | None

    expected_calibration_error: float

    total_decision_regret: float
    average_decision_regret: float

    cost_weighted_accuracy: float

    confusion_matrix: dict[
        str,
        dict[str, int],
    ]

    per_class: dict[
        str,
        ClassMetrics,
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples":
                self.total_samples,

            "correct_predictions":
                self.correct_predictions,

            "incorrect_predictions":
                self.incorrect_predictions,

            "accuracy":
                self.accuracy,

            "macro_precision":
                self.macro_precision,

            "macro_recall":
                self.macro_recall,

            "macro_f1":
                self.macro_f1,

            "weighted_precision":
                self.weighted_precision,

            "weighted_recall":
                self.weighted_recall,

            "weighted_f1":
                self.weighted_f1,

            "repair_accuracy":
                self.repair_accuracy,

            "replacement_accuracy":
                self.replacement_accuracy,

            "repair_vs_replace_accuracy":
                self.repair_vs_replace_accuracy,

            "average_confidence":
                self.average_confidence,

            "correct_average_confidence":
                self.correct_average_confidence,

            "incorrect_average_confidence":
                self.incorrect_average_confidence,

            "expected_calibration_error":
                self.expected_calibration_error,

            "total_decision_regret":
                self.total_decision_regret,

            "average_decision_regret":
                self.average_decision_regret,

            "cost_weighted_accuracy":
                self.cost_weighted_accuracy,

            "confusion_matrix":
                self.confusion_matrix,

            "per_class": {
                key: value.to_dict()
                for key, value
                in self.per_class.items()
            },

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

    if not isfinite(value):
        raise ValueError(
            "confidence must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "confidence must be between "
            "0 and 1."
        )

    return value


def _safe_cost(
    value: float | None,
) -> float:

    if value is None:
        return 0.0

    value = float(value)

    if not isfinite(value):
        raise ValueError(
            "Cost values must be finite."
        )

    return max(
        0.0,
        value,
    )


# =========================================================
# Confusion matrix
# =========================================================

def build_confusion_matrix(
    records: Iterable[
        DecisionEvaluationRecord
    ],
) -> dict[str, dict[str, int]]:

    matrix = {
        actual: {
            predicted: 0
            for predicted
            in DECISION_LABELS
        }
        for actual
        in DECISION_LABELS
    }

    for record in records:
        actual = (
            record.actual_decision.value
        )

        predicted = (
            record.predicted_decision.value
        )

        matrix[actual][predicted] += 1

    return matrix


# =========================================================
# Per-class metrics
# =========================================================

def calculate_class_metrics(
    *,
    label: str,
    matrix: dict[
        str,
        dict[str, int],
    ],
) -> ClassMetrics:

    total = sum(
        sum(row.values())
        for row in matrix.values()
    )

    tp = matrix[label][label]

    fp = sum(
        matrix[actual][label]
        for actual in DECISION_LABELS
        if actual != label
    )

    fn = sum(
        matrix[label][predicted]
        for predicted in DECISION_LABELS
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
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    support = tp + fn

    return ClassMetrics(
        label=label,

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

        support=support,
    )


# =========================================================
# Specific action accuracy
# =========================================================

def calculate_action_accuracy(
    records: list[
        DecisionEvaluationRecord
    ],
    decision: RepairReplacementDecision,
) -> float | None:

    relevant = [
        record
        for record in records
        if record.actual_decision
        == decision
    ]

    if not relevant:
        return None

    correct = sum(
        record.predicted_decision
        == record.actual_decision
        for record in relevant
    )

    return round(
        correct / len(relevant),
        4,
    )


# =========================================================
# Repair vs replacement accuracy
# =========================================================

def calculate_repair_replace_accuracy(
    records: list[
        DecisionEvaluationRecord
    ],
) -> float | None:
    """
    Evaluate only cases where the actual and predicted
    decision are in the repair/replacement decision space.

    Useful for the key business question:

        Should this asset be repaired or replaced?
    """

    allowed = {
        RepairReplacementDecision.REPAIR,
        RepairReplacementDecision.REPLACE,
    }

    relevant = [
        record
        for record in records
        if record.actual_decision
        in allowed
    ]

    if not relevant:
        return None

    correct = sum(
        record.predicted_decision
        == record.actual_decision
        for record in relevant
    )

    return round(
        correct / len(relevant),
        4,
    )


# =========================================================
# Decision regret
# =========================================================

def calculate_decision_regret(
    record: DecisionEvaluationRecord,
) -> float:
    """
    Decision regret measures the additional cost caused
    by the AI decision relative to the optimal action.

    If realized cost is available:

        regret =
            realized cost - optimal action cost

    Otherwise:

        regret =
            predicted action cost - optimal action cost

    For a wrong decision, estimated failure consequence
    can also be included.
    """

    optimal = _safe_cost(
        record.optimal_action_cost
    )

    if (
        record.realized_total_cost
        is not None
    ):
        selected_cost = _safe_cost(
            record.realized_total_cost
        )
    else:
        selected_cost = _safe_cost(
            record.predicted_action_cost
        )

        if (
            record.predicted_decision
            != record.actual_decision
        ):
            selected_cost += _safe_cost(
                record.failure_consequence_cost
            )

    return round(
        max(
            0.0,
            selected_cost - optimal,
        ),
        2,
    )


# =========================================================
# Cost-weighted accuracy
# =========================================================

def calculate_cost_weighted_accuracy(
    records: list[
        DecisionEvaluationRecord
    ],
) -> float:
    """
    Gives more importance to expensive mistakes.

    A ₹1,000,000 incorrect replacement/failure decision
    therefore matters more than a ₹1,000 minor mistake.
    """

    if not records:
        return 0.0

    total_exposure = 0.0
    total_regret = 0.0

    for record in records:
        exposure = max(
            _safe_cost(
                record.optimal_action_cost
            ),
            _safe_cost(
                record.failure_consequence_cost
            ),
            1.0,
        )

        total_exposure += exposure

        total_regret += min(
            calculate_decision_regret(
                record
            ),
            exposure,
        )

    if total_exposure <= 0:
        return 0.0

    score = (
        1.0
        - (
            total_regret
            / total_exposure
        )
    )

    return round(
        max(
            0.0,
            min(
                1.0,
                score,
            ),
        ),
        4,
    )


# =========================================================
# Confidence calibration
# =========================================================

def calculate_ece(
    records: list[
        DecisionEvaluationRecord
    ],
    *,
    bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    A well-calibrated model making predictions with
    confidence ~0.80 should be correct roughly 80%
    of the time.
    """

    if not records:
        return 0.0

    if bins < 1:
        raise ValueError(
            "bins must be >= 1."
        )

    bucket_data: list[
        list[
            DecisionEvaluationRecord
        ]
    ] = [
        []
        for _ in range(bins)
    ]

    for record in records:
        confidence = (
            _validate_confidence(
                record.confidence
            )
        )

        index = min(
            int(confidence * bins),
            bins - 1,
        )

        bucket_data[index].append(
            record
        )

    total = len(records)

    ece = 0.0

    for bucket in bucket_data:
        if not bucket:
            continue

        bucket_accuracy = mean(
            1.0
            if record.predicted_decision
            == record.actual_decision
            else 0.0
            for record in bucket
        )

        bucket_confidence = mean(
            _validate_confidence(
                record.confidence
            )
            for record in bucket
        )

        weight = (
            len(bucket)
            / total
        )

        ece += (
            weight
            * abs(
                bucket_accuracy
                - bucket_confidence
            )
        )

    return round(
        ece,
        4,
    )


# =========================================================
# Main evaluator
# =========================================================

def evaluate_repair_replacement_accuracy(
    records: list[
        DecisionEvaluationRecord
    ],
) -> RepairReplacementAccuracyResult:

    if not records:
        return RepairReplacementAccuracyResult(
            total_samples=0,

            correct_predictions=0,
            incorrect_predictions=0,

            accuracy=0.0,

            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,

            weighted_precision=0.0,
            weighted_recall=0.0,
            weighted_f1=0.0,

            repair_accuracy=None,
            replacement_accuracy=None,

            repair_vs_replace_accuracy=None,

            average_confidence=0.0,
            correct_average_confidence=None,
            incorrect_average_confidence=None,

            expected_calibration_error=0.0,

            total_decision_regret=0.0,
            average_decision_regret=0.0,

            cost_weighted_accuracy=0.0,

            confusion_matrix={},
            per_class={},
            errors=[],
        )

    for record in records:
        _validate_confidence(
            record.confidence
        )

    matrix = build_confusion_matrix(
        records
    )

    per_class = {
        label: calculate_class_metrics(
            label=label,
            matrix=matrix,
        )
        for label in DECISION_LABELS
    }

    correct_records = [
        record
        for record in records
        if record.predicted_decision
        == record.actual_decision
    ]

    incorrect_records = [
        record
        for record in records
        if record.predicted_decision
        != record.actual_decision
    ]

    total = len(records)

    correct = len(
        correct_records
    )

    incorrect = (
        total - correct
    )

    accuracy = (
        correct / total
    )

    supported_metrics = [
        metric
        for metric in per_class.values()
        if metric.support > 0
    ]

    if supported_metrics:
        macro_precision = mean(
            metric.precision
            for metric
            in supported_metrics
        )

        macro_recall = mean(
            metric.recall
            for metric
            in supported_metrics
        )

        macro_f1 = mean(
            metric.f1_score
            for metric
            in supported_metrics
        )

        weighted_precision = (
            sum(
                metric.precision
                * metric.support
                for metric
                in supported_metrics
            )
            / total
        )

        weighted_recall = (
            sum(
                metric.recall
                * metric.support
                for metric
                in supported_metrics
            )
            / total
        )

        weighted_f1 = (
            sum(
                metric.f1_score
                * metric.support
                for metric
                in supported_metrics
            )
            / total
        )

    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    confidences = [
        record.confidence
        for record in records
    ]

    correct_confidences = [
        record.confidence
        for record in correct_records
    ]

    incorrect_confidences = [
        record.confidence
        for record in incorrect_records
    ]

    regrets = [
        calculate_decision_regret(
            record
        )
        for record in records
    ]

    total_regret = sum(
        regrets
    )

    errors = [
        {
            "asset_id":
                record.asset_id,

            "predicted":
                record.predicted_decision.value,

            "actual":
                record.actual_decision.value,

            "confidence":
                record.confidence,

            "decision_regret":
                calculate_decision_regret(
                    record
                ),

            "prediction_id":
                record.prediction_id,

            "recommendation_id":
                record.recommendation_id,

            "work_order_id":
                record.work_order_id,
        }
        for record in incorrect_records
    ]

    return RepairReplacementAccuracyResult(
        total_samples=total,

        correct_predictions=correct,
        incorrect_predictions=incorrect,

        accuracy=round(
            accuracy,
            4,
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

        weighted_precision=round(
            weighted_precision,
            4,
        ),

        weighted_recall=round(
            weighted_recall,
            4,
        ),

        weighted_f1=round(
            weighted_f1,
            4,
        ),

        repair_accuracy=(
            calculate_action_accuracy(
                records,
                RepairReplacementDecision.REPAIR,
            )
        ),

        replacement_accuracy=(
            calculate_action_accuracy(
                records,
                RepairReplacementDecision.REPLACE,
            )
        ),

        repair_vs_replace_accuracy=(
            calculate_repair_replace_accuracy(
                records
            )
        ),

        average_confidence=round(
            mean(confidences),
            4,
        ),

        correct_average_confidence=(
            round(
                mean(
                    correct_confidences
                ),
                4,
            )
            if correct_confidences
            else None
        ),

        incorrect_average_confidence=(
            round(
                mean(
                    incorrect_confidences
                ),
                4,
            )
            if incorrect_confidences
            else None
        ),

        expected_calibration_error=(
            calculate_ece(
                records
            )
        ),

        total_decision_regret=round(
            total_regret,
            2,
        ),

        average_decision_regret=round(
            total_regret / total,
            2,
        ),

        cost_weighted_accuracy=(
            calculate_cost_weighted_accuracy(
                records
            )
        ),

        confusion_matrix=matrix,

        per_class=per_class,

        errors=errors,
    )


# =========================================================
# Threshold evaluation
# =========================================================

def evaluate_confidence_threshold(
    records: list[
        DecisionEvaluationRecord
    ],
    *,
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate only recommendations whose confidence
    meets a deployment threshold.

    Example:
        Only automatically execute recommendations
        when confidence >= 0.85.
    """

    _validate_confidence(
        threshold
    )

    accepted = [
        record
        for record in records
        if record.confidence
        >= threshold
    ]

    rejected = [
        record
        for record in records
        if record.confidence
        < threshold
    ]

    if accepted:
        correct = sum(
            record.predicted_decision
            == record.actual_decision
            for record in accepted
        )

        accuracy = (
            correct
            / len(accepted)
        )
    else:
        accuracy = 0.0

    coverage = (
        len(accepted)
        / len(records)
        if records
        else 0.0
    )

    return {
        "threshold":
            threshold,

        "total":
            len(records),

        "accepted":
            len(accepted),

        "rejected":
            len(rejected),

        "coverage":
            round(
                coverage,
                4,
            ),

        "accepted_accuracy":
            round(
                accuracy,
                4,
            ),
    }


# =========================================================
# Evaluator service
# =========================================================

class RepairReplacementAccuracyEvaluator:

    def evaluate(
        self,
        records: list[
            DecisionEvaluationRecord
        ],
    ) -> RepairReplacementAccuracyResult:

        return (
            evaluate_repair_replacement_accuracy(
                records
            )
        )

    def evaluate_threshold(
        self,
        records: list[
            DecisionEvaluationRecord
        ],
        threshold: float,
    ) -> dict[str, Any]:

        return (
            evaluate_confidence_threshold(
                records,
                threshold=threshold,
            )
        )


repair_replacement_accuracy_evaluator = (
    RepairReplacementAccuracyEvaluator()
)