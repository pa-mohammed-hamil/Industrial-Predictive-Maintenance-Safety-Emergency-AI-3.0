# app/evaluation/rul_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import exp, isfinite, sqrt
from statistics import mean, median
from typing import Any, Iterable, Sequence


# =========================================================
# Enums
# =========================================================

class RULUnit(str, Enum):
    HOURS = "hours"
    DAYS = "days"
    CYCLES = "cycles"


class RULRiskLevel(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class RULErrorDirection(str, Enum):
    EARLY = "early"
    EXACT = "exact"
    LATE = "late"


# =========================================================
# Prediction / ground truth
# =========================================================

@dataclass(slots=True)
class RULPrediction:
    predicted_rul: float

    confidence: float = 1.0

    lower_bound: float | None = None
    upper_bound: float | None = None

    risk_level: RULRiskLevel | None = None

    model_name: str | None = None
    model_version: str | None = None

    contributing_features: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class RULGroundTruth:
    actual_rul: float

    risk_level: RULRiskLevel | None = None


@dataclass(slots=True)
class RULEvaluationSample:
    sample_id: str
    asset_id: str

    prediction: RULPrediction
    ground_truth: RULGroundTruth

    unit: RULUnit = RULUnit.HOURS

    timestamp_ms: float | None = None

    site_id: str | None = None
    sensor_id: str | None = None

    prediction_id: str | None = None
    failure_id: str | None = None
    work_order_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Metric structures
# =========================================================

@dataclass(slots=True)
class RULGroupMetrics:
    samples: int

    mae: float
    mse: float
    rmse: float

    median_absolute_error: float

    mape: float | None
    smape: float

    bias: float
    r2: float | None

    early_predictions: int
    late_predictions: int
    exact_predictions: int

    early_prediction_rate: float
    late_prediction_rate: float

    mean_early_error: float | None
    mean_late_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RULEvaluationResult:
    total_samples: int

    mae: float
    mse: float
    rmse: float

    median_absolute_error: float

    mape: float | None
    smape: float

    bias: float
    r2: float | None

    normalized_mae: float | None
    normalized_rmse: float | None

    early_predictions: int
    late_predictions: int
    exact_predictions: int

    early_prediction_rate: float
    late_prediction_rate: float

    mean_early_error: float | None
    mean_late_error: float | None

    asymmetric_score: float

    within_5_percent: float
    within_10_percent: float
    within_20_percent: float

    within_absolute_tolerance: float

    critical_rul_mae: float | None
    critical_rul_rmse: float | None

    critical_rul_detection_recall: float | None

    risk_level_accuracy: float | None

    interval_coverage: float | None
    mean_interval_width: float | None

    average_confidence: float

    per_asset: dict[str, RULGroupMetrics]
    per_model: dict[str, RULGroupMetrics]

    largest_errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_asset"] = {
            key: value.to_dict()
            for key, value in self.per_asset.items()
        }

        result["per_model"] = {
            key: value.to_dict()
            for key, value in self.per_model.items()
        }

        return result


# =========================================================
# Validation
# =========================================================

def _finite(
    value: float,
    name: str,
) -> float:

    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


def _non_negative(
    value: float,
    name: str,
) -> float:

    value = _finite(
        value,
        name,
    )

    if value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


def _probability(
    value: float,
    name: str,
) -> float:

    value = _finite(
        value,
        name,
    )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def validate_sample(
    sample: RULEvaluationSample,
) -> None:

    _non_negative(
        sample.prediction.predicted_rul,
        "predicted_rul",
    )

    _non_negative(
        sample.ground_truth.actual_rul,
        "actual_rul",
    )

    _probability(
        sample.prediction.confidence,
        "confidence",
    )

    lower = sample.prediction.lower_bound
    upper = sample.prediction.upper_bound

    if lower is not None:
        _non_negative(
            lower,
            "lower_bound",
        )

    if upper is not None:
        _non_negative(
            upper,
            "upper_bound",
        )

    if (
        lower is not None
        and upper is not None
        and lower > upper
    ):
        raise ValueError(
            "lower_bound cannot be greater "
            "than upper_bound."
        )


def _validate_arrays(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> None:

    if len(actual) != len(predicted):
        raise ValueError(
            "actual and predicted must "
            "have equal lengths."
        )

    for value in actual:
        _non_negative(
            value,
            "actual_rul",
        )

    for value in predicted:
        _non_negative(
            value,
            "predicted_rul",
        )


# =========================================================
# Error helpers
# =========================================================

def calculate_rul_error(
    actual_rul: float,
    predicted_rul: float,
) -> float:
    """
    Signed error:

        predicted - actual

    Positive:
        model overestimates remaining life.
        This is a late / optimistic prediction.

    Negative:
        model underestimates remaining life.
        This is an early / conservative prediction.
    """

    return (
        float(predicted_rul)
        - float(actual_rul)
    )


def classify_error_direction(
    actual_rul: float,
    predicted_rul: float,
    *,
    epsilon: float = 1e-9,
) -> RULErrorDirection:

    error = calculate_rul_error(
        actual_rul,
        predicted_rul,
    )

    if error > epsilon:
        return RULErrorDirection.LATE

    if error < -epsilon:
        return RULErrorDirection.EARLY

    return RULErrorDirection.EXACT


# =========================================================
# MAE / MSE / RMSE
# =========================================================

def calculate_mae(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:

    _validate_arrays(
        actual,
        predicted,
    )

    if not actual:
        return 0.0

    return round(
        mean(
            abs(predicted_value - actual_value)
            for actual_value, predicted_value
            in zip(actual, predicted)
        ),
        6,
    )


def calculate_mse(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:

    _validate_arrays(
        actual,
        predicted,
    )

    if not actual:
        return 0.0

    return round(
        mean(
            (predicted_value - actual_value) ** 2
            for actual_value, predicted_value
            in zip(actual, predicted)
        ),
        6,
    )


def calculate_rmse(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:

    return round(
        sqrt(
            calculate_mse(
                actual,
                predicted,
            )
        ),
        6,
    )


# =========================================================
# MAPE / SMAPE
# =========================================================

def calculate_mape(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> float | None:

    _validate_arrays(
        actual,
        predicted,
    )

    errors: list[float] = []

    for actual_value, predicted_value in zip(
        actual,
        predicted,
    ):
        if abs(actual_value) <= epsilon:
            continue

        errors.append(
            abs(
                (
                    predicted_value
                    - actual_value
                )
                / actual_value
            )
            * 100.0
        )

    if not errors:
        return None

    return round(
        mean(errors),
        4,
    )


def calculate_smape(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> float:

    _validate_arrays(
        actual,
        predicted,
    )

    if not actual:
        return 0.0

    values: list[float] = []

    for actual_value, predicted_value in zip(
        actual,
        predicted,
    ):
        denominator = (
            abs(actual_value)
            + abs(predicted_value)
        )

        if denominator <= epsilon:
            values.append(0.0)
            continue

        values.append(
            (
                2.0
                * abs(
                    predicted_value
                    - actual_value
                )
                / denominator
            )
            * 100.0
        )

    return round(
        mean(values),
        4,
    )


# =========================================================
# Bias
# =========================================================

def calculate_bias(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:

    _validate_arrays(
        actual,
        predicted,
    )

    if not actual:
        return 0.0

    return round(
        mean(
            predicted_value
            - actual_value
            for actual_value, predicted_value
            in zip(actual, predicted)
        ),
        6,
    )


# =========================================================
# R2
# =========================================================

def calculate_r2(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float | None:

    _validate_arrays(
        actual,
        predicted,
    )

    if len(actual) < 2:
        return None

    actual_mean = mean(actual)

    total_variance = sum(
        (
            value
            - actual_mean
        ) ** 2
        for value in actual
    )

    if total_variance == 0:
        return None

    residual = sum(
        (
            actual_value
            - predicted_value
        ) ** 2
        for actual_value, predicted_value
        in zip(actual, predicted)
    )

    return round(
        1.0
        - (
            residual
            / total_variance
        ),
        6,
    )


# =========================================================
# Normalized errors
# =========================================================

def calculate_normalized_error(
    error: float,
    actual: Sequence[float],
) -> float | None:

    if not actual:
        return None

    value_range = (
        max(actual)
        - min(actual)
    )

    if value_range <= 0:
        return None

    return round(
        float(error)
        / value_range,
        6,
    )


# =========================================================
# Early / late prediction analysis
# =========================================================

def calculate_direction_metrics(
    samples: Sequence[
        RULEvaluationSample
    ],
) -> dict[str, Any]:

    early_errors: list[float] = []
    late_errors: list[float] = []

    exact = 0

    for sample in samples:
        actual = (
            sample.ground_truth.actual_rul
        )

        predicted = (
            sample.prediction.predicted_rul
        )

        error = calculate_rul_error(
            actual,
            predicted,
        )

        direction = (
            classify_error_direction(
                actual,
                predicted,
            )
        )

        if (
            direction
            == RULErrorDirection.EARLY
        ):
            early_errors.append(
                abs(error)
            )

        elif (
            direction
            == RULErrorDirection.LATE
        ):
            late_errors.append(
                error
            )

        else:
            exact += 1

    total = len(samples)

    return {
        "early_predictions":
            len(early_errors),

        "late_predictions":
            len(late_errors),

        "exact_predictions":
            exact,

        "early_prediction_rate":
            round(
                len(early_errors) / total,
                4,
            )
            if total
            else 0.0,

        "late_prediction_rate":
            round(
                len(late_errors) / total,
                4,
            )
            if total
            else 0.0,

        "mean_early_error":
            round(
                mean(early_errors),
                6,
            )
            if early_errors
            else None,

        "mean_late_error":
            round(
                mean(late_errors),
                6,
            )
            if late_errors
            else None,
    }


# =========================================================
# Asymmetric RUL score
# =========================================================

def calculate_asymmetric_score(
    samples: Sequence[
        RULEvaluationSample
    ],
    *,
    early_scale: float = 13.0,
    late_scale: float = 10.0,
) -> float:
    """
    Penalizes optimistic / late RUL predictions more
    strongly than conservative / early predictions.

    Error = predicted RUL - actual RUL.

    predicted > actual:
        asset fails earlier than predicted.
        Higher operational/safety risk.

    predicted < actual:
        maintenance may happen too early.
    """

    early_scale = _non_negative(
        early_scale,
        "early_scale",
    )

    late_scale = _non_negative(
        late_scale,
        "late_scale",
    )

    if (
        early_scale == 0
        or late_scale == 0
    ):
        raise ValueError(
            "Penalty scales must be > 0."
        )

    score = 0.0

    for sample in samples:
        error = calculate_rul_error(
            sample.ground_truth.actual_rul,
            sample.prediction.predicted_rul,
        )

        if error < 0:
            score += (
                exp(
                    abs(error)
                    / early_scale
                )
                - 1.0
            )

        else:
            score += (
                exp(
                    error
                    / late_scale
                )
                - 1.0
            )

    return round(
        score,
        6,
    )


# =========================================================
# Tolerance accuracy
# =========================================================

def calculate_percentage_tolerance_accuracy(
    samples: Sequence[
        RULEvaluationSample
    ],
    *,
    tolerance_percent: float,
    zero_actual_absolute_tolerance: float = 0.0,
) -> float:

    if tolerance_percent < 0:
        raise ValueError(
            "tolerance_percent cannot be negative."
        )

    if not samples:
        return 0.0

    correct = 0

    for sample in samples:
        actual = (
            sample.ground_truth.actual_rul
        )

        predicted = (
            sample.prediction.predicted_rul
        )

        error = abs(
            predicted - actual
        )

        if actual == 0:
            within = (
                error
                <= zero_actual_absolute_tolerance
            )

        else:
            within = (
                error
                <= (
                    abs(actual)
                    * tolerance_percent
                    / 100.0
                )
            )

        if within:
            correct += 1

    return round(
        correct / len(samples),
        4,
    )


def calculate_absolute_tolerance_accuracy(
    samples: Sequence[
        RULEvaluationSample
    ],
    *,
    tolerance: float,
) -> float:

    tolerance = _non_negative(
        tolerance,
        "tolerance",
    )

    if not samples:
        return 0.0

    correct = sum(
        abs(
            sample.prediction.predicted_rul
            - sample.ground_truth.actual_rul
        )
        <= tolerance
        for sample in samples
    )

    return round(
        correct / len(samples),
        4,
    )


# =========================================================
# Critical RUL evaluation
# =========================================================

def evaluate_critical_rul(
    samples: Sequence[
        RULEvaluationSample
    ],
    *,
    critical_threshold: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
]:

    critical_threshold = _non_negative(
        critical_threshold,
        "critical_threshold",
    )

    critical_samples = [
        sample
        for sample in samples
        if (
            sample.ground_truth.actual_rul
            <= critical_threshold
        )
    ]

    if not critical_samples:
        return (
            None,
            None,
            None,
        )

    actual = [
        sample.ground_truth.actual_rul
        for sample in critical_samples
    ]

    predicted = [
        sample.prediction.predicted_rul
        for sample in critical_samples
    ]

    detected = sum(
        sample.prediction.predicted_rul
        <= critical_threshold
        for sample in critical_samples
    )

    return (
        calculate_mae(
            actual,
            predicted,
        ),
        calculate_rmse(
            actual,
            predicted,
        ),
        round(
            detected
            / len(critical_samples),
            4,
        ),
    )


# =========================================================
# Risk classification
# =========================================================

def calculate_risk_level_accuracy(
    samples: Sequence[
        RULEvaluationSample
    ],
) -> float | None:

    comparable = [
        sample
        for sample in samples
        if (
            sample.prediction.risk_level
            is not None
            and
            sample.ground_truth.risk_level
            is not None
        )
    ]

    if not comparable:
        return None

    correct = sum(
        sample.prediction.risk_level
        == sample.ground_truth.risk_level
        for sample in comparable
    )

    return round(
        correct / len(comparable),
        4,
    )


# =========================================================
# Prediction interval metrics
# =========================================================

def calculate_interval_metrics(
    samples: Sequence[
        RULEvaluationSample
    ],
) -> tuple[
    float | None,
    float | None,
]:

    valid = [
        sample
        for sample in samples
        if (
            sample.prediction.lower_bound
            is not None
            and
            sample.prediction.upper_bound
            is not None
        )
    ]

    if not valid:
        return (
            None,
            None,
        )

    covered = 0
    widths: list[float] = []

    for sample in valid:
        lower = float(
            sample.prediction.lower_bound
        )

        upper = float(
            sample.prediction.upper_bound
        )

        actual = (
            sample.ground_truth.actual_rul
        )

        if lower <= actual <= upper:
            covered += 1

        widths.append(
            upper - lower
        )

    return (
        round(
            covered / len(valid),
            4,
        ),
        round(
            mean(widths),
            6,
        ),
    )


# =========================================================
# Group metrics
# =========================================================

def calculate_group_metrics(
    samples: Sequence[
        RULEvaluationSample
    ],
) -> RULGroupMetrics:

    if not samples:
        return RULGroupMetrics(
            samples=0,
            mae=0.0,
            mse=0.0,
            rmse=0.0,
            median_absolute_error=0.0,
            mape=None,
            smape=0.0,
            bias=0.0,
            r2=None,
            early_predictions=0,
            late_predictions=0,
            exact_predictions=0,
            early_prediction_rate=0.0,
            late_prediction_rate=0.0,
            mean_early_error=None,
            mean_late_error=None,
        )

    actual = [
        sample.ground_truth.actual_rul
        for sample in samples
    ]

    predicted = [
        sample.prediction.predicted_rul
        for sample in samples
    ]

    absolute_errors = [
        abs(p - a)
        for a, p
        in zip(actual, predicted)
    ]

    directions = (
        calculate_direction_metrics(
            samples
        )
    )

    return RULGroupMetrics(
        samples=len(samples),

        mae=calculate_mae(
            actual,
            predicted,
        ),

        mse=calculate_mse(
            actual,
            predicted,
        ),

        rmse=calculate_rmse(
            actual,
            predicted,
        ),

        median_absolute_error=round(
            median(absolute_errors),
            6,
        ),

        mape=calculate_mape(
            actual,
            predicted,
        ),

        smape=calculate_smape(
            actual,
            predicted,
        ),

        bias=calculate_bias(
            actual,
            predicted,
        ),

        r2=calculate_r2(
            actual,
            predicted,
        ),

        early_predictions=(
            directions[
                "early_predictions"
            ]
        ),

        late_predictions=(
            directions[
                "late_predictions"
            ]
        ),

        exact_predictions=(
            directions[
                "exact_predictions"
            ]
        ),

        early_prediction_rate=(
            directions[
                "early_prediction_rate"
            ]
        ),

        late_prediction_rate=(
            directions[
                "late_prediction_rate"
            ]
        ),

        mean_early_error=(
            directions[
                "mean_early_error"
            ]
        ),

        mean_late_error=(
            directions[
                "mean_late_error"
            ]
        ),
    )


# =========================================================
# Per-asset metrics
# =========================================================

def calculate_per_asset_metrics(
    samples: Sequence[
        RULEvaluationSample
    ],
) -> dict[str, RULGroupMetrics]:

    groups: dict[
        str,
        list[RULEvaluationSample],
    ] = {}

    for sample in samples:
        groups.setdefault(
            sample.asset_id,
            [],
        ).append(sample)

    return {
        asset_id:
            calculate_group_metrics(
                group
            )
        for asset_id, group
        in groups.items()
    }


# =========================================================
# Per-model metrics
# =========================================================

def calculate_per_model_metrics(
    samples: Sequence[
        RULEvaluationSample
    ],
) -> dict[str, RULGroupMetrics]:

    groups: dict[
        str,
        list[RULEvaluationSample],
    ] = {}

    for sample in samples:
        model_name = (
            sample.prediction.model_name
            or "unknown"
        )

        groups.setdefault(
            model_name,
            [],
        ).append(sample)

    return {
        model:
            calculate_group_metrics(
                group
            )
        for model, group
        in groups.items()
    }


# =========================================================
# Largest errors
# =========================================================

def get_largest_errors(
    samples: Sequence[
        RULEvaluationSample
    ],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:

    if limit <= 0:
        return []

    ordered = sorted(
        samples,
        key=lambda sample: abs(
            sample.prediction.predicted_rul
            - sample.ground_truth.actual_rul
        ),
        reverse=True,
    )

    errors: list[dict[str, Any]] = []

    for sample in ordered[:limit]:
        actual = (
            sample.ground_truth.actual_rul
        )

        predicted = (
            sample.prediction.predicted_rul
        )

        signed_error = (
            predicted - actual
        )

        errors.append(
            {
                "sample_id":
                    sample.sample_id,

                "asset_id":
                    sample.asset_id,

                "site_id":
                    sample.site_id,

                "sensor_id":
                    sample.sensor_id,

                "unit":
                    sample.unit.value,

                "actual_rul":
                    actual,

                "predicted_rul":
                    predicted,

                "signed_error":
                    round(
                        signed_error,
                        6,
                    ),

                "absolute_error":
                    round(
                        abs(signed_error),
                        6,
                    ),

                "direction":
                    classify_error_direction(
                        actual,
                        predicted,
                    ).value,

                "confidence":
                    sample.prediction.confidence,

                "model_name":
                    sample.prediction.model_name,

                "model_version":
                    sample.prediction.model_version,

                "prediction_id":
                    sample.prediction_id,

                "failure_id":
                    sample.failure_id,

                "work_order_id":
                    sample.work_order_id,
            }
        )

    return errors


# =========================================================
# Main evaluation
# =========================================================

def evaluate_rul(
    samples: list[
        RULEvaluationSample
    ],
    *,
    critical_threshold: float = 24.0,
    absolute_tolerance: float = 12.0,
    largest_error_limit: int = 20,
) -> RULEvaluationResult:

    if not samples:
        return RULEvaluationResult(
            total_samples=0,

            mae=0.0,
            mse=0.0,
            rmse=0.0,

            median_absolute_error=0.0,

            mape=None,
            smape=0.0,

            bias=0.0,
            r2=None,

            normalized_mae=None,
            normalized_rmse=None,

            early_predictions=0,
            late_predictions=0,
            exact_predictions=0,

            early_prediction_rate=0.0,
            late_prediction_rate=0.0,

            mean_early_error=None,
            mean_late_error=None,

            asymmetric_score=0.0,

            within_5_percent=0.0,
            within_10_percent=0.0,
            within_20_percent=0.0,

            within_absolute_tolerance=0.0,

            critical_rul_mae=None,
            critical_rul_rmse=None,
            critical_rul_detection_recall=None,

            risk_level_accuracy=None,

            interval_coverage=None,
            mean_interval_width=None,

            average_confidence=0.0,

            per_asset={},
            per_model={},

            largest_errors=[],
        )

    for sample in samples:
        validate_sample(sample)

    actual = [
        sample.ground_truth.actual_rul
        for sample in samples
    ]

    predicted = [
        sample.prediction.predicted_rul
        for sample in samples
    ]

    absolute_errors = [
        abs(p - a)
        for a, p
        in zip(actual, predicted)
    ]

    mae = calculate_mae(
        actual,
        predicted,
    )

    mse = calculate_mse(
        actual,
        predicted,
    )

    rmse = calculate_rmse(
        actual,
        predicted,
    )

    directions = (
        calculate_direction_metrics(
            samples
        )
    )

    (
        critical_mae,
        critical_rmse,
        critical_recall,
    ) = evaluate_critical_rul(
        samples,
        critical_threshold=(
            critical_threshold
        ),
    )

    (
        interval_coverage,
        interval_width,
    ) = calculate_interval_metrics(
        samples
    )

    return RULEvaluationResult(
        total_samples=len(samples),

        mae=mae,
        mse=mse,
        rmse=rmse,

        median_absolute_error=round(
            median(absolute_errors),
            6,
        ),

        mape=calculate_mape(
            actual,
            predicted,
        ),

        smape=calculate_smape(
            actual,
            predicted,
        ),

        bias=calculate_bias(
            actual,
            predicted,
        ),

        r2=calculate_r2(
            actual,
            predicted,
        ),

        normalized_mae=(
            calculate_normalized_error(
                mae,
                actual,
            )
        ),

        normalized_rmse=(
            calculate_normalized_error(
                rmse,
                actual,
            )
        ),

        early_predictions=(
            directions[
                "early_predictions"
            ]
        ),

        late_predictions=(
            directions[
                "late_predictions"
            ]
        ),

        exact_predictions=(
            directions[
                "exact_predictions"
            ]
        ),

        early_prediction_rate=(
            directions[
                "early_prediction_rate"
            ]
        ),

        late_prediction_rate=(
            directions[
                "late_prediction_rate"
            ]
        ),

        mean_early_error=(
            directions[
                "mean_early_error"
            ]
        ),

        mean_late_error=(
            directions[
                "mean_late_error"
            ]
        ),

        asymmetric_score=(
            calculate_asymmetric_score(
                samples
            )
        ),

        within_5_percent=(
            calculate_percentage_tolerance_accuracy(
                samples,
                tolerance_percent=5.0,
            )
        ),

        within_10_percent=(
            calculate_percentage_tolerance_accuracy(
                samples,
                tolerance_percent=10.0,
            )
        ),

        within_20_percent=(
            calculate_percentage_tolerance_accuracy(
                samples,
                tolerance_percent=20.0,
            )
        ),

        within_absolute_tolerance=(
            calculate_absolute_tolerance_accuracy(
                samples,
                tolerance=absolute_tolerance,
            )
        ),

        critical_rul_mae=(
            critical_mae
        ),

        critical_rul_rmse=(
            critical_rmse
        ),

        critical_rul_detection_recall=(
            critical_recall
        ),

        risk_level_accuracy=(
            calculate_risk_level_accuracy(
                samples
            )
        ),

        interval_coverage=(
            interval_coverage
        ),

        mean_interval_width=(
            interval_width
        ),

        average_confidence=round(
            mean(
                sample.prediction.confidence
                for sample in samples
            ),
            4,
        ),

        per_asset=(
            calculate_per_asset_metrics(
                samples
            )
        ),

        per_model=(
            calculate_per_model_metrics(
                samples
            )
        ),

        largest_errors=(
            get_largest_errors(
                samples,
                limit=largest_error_limit,
            )
        ),
    )


# =========================================================
# Compare models
# =========================================================

def compare_rul_models(
    model_samples: dict[
        str,
        list[RULEvaluationSample],
    ],
    *,
    critical_threshold: float = 24.0,
) -> list[dict[str, Any]]:

    comparisons: list[
        dict[str, Any]
    ] = []

    for model_name, samples in (
        model_samples.items()
    ):
        result = evaluate_rul(
            samples,
            critical_threshold=(
                critical_threshold
            ),
            largest_error_limit=5,
        )

        comparisons.append(
            {
                "model":
                    model_name,

                "samples":
                    result.total_samples,

                "mae":
                    result.mae,

                "rmse":
                    result.rmse,

                "mape":
                    result.mape,

                "smape":
                    result.smape,

                "r2":
                    result.r2,

                "bias":
                    result.bias,

                "late_prediction_rate":
                    result.late_prediction_rate,

                "critical_rul_recall":
                    result
                    .critical_rul_detection_recall,

                "within_10_percent":
                    result.within_10_percent,

                "asymmetric_score":
                    result.asymmetric_score,
            }
        )

    # Lower asymmetric risk score first,
    # then lower RMSE.
    comparisons.sort(
        key=lambda item: (
            item["asymmetric_score"],
            item["rmse"],
        )
    )

    return comparisons


# =========================================================
# Rolling evaluation
# =========================================================

def evaluate_rul_windows(
    samples: list[
        RULEvaluationSample
    ],
    *,
    window_size: int,
    critical_threshold: float = 24.0,
) -> list[dict[str, Any]]:

    if window_size < 1:
        raise ValueError(
            "window_size must be >= 1."
        )

    if not samples:
        return []

    ordered = sorted(
        samples,
        key=lambda sample: (
            float(sample.timestamp_ms)
            if sample.timestamp_ms
            is not None
            else 0.0
        ),
    )

    results: list[
        dict[str, Any]
    ] = []

    for start in range(
        0,
        len(ordered),
        window_size,
    ):
        window = ordered[
            start:
            start + window_size
        ]

        evaluation = evaluate_rul(
            window,
            critical_threshold=(
                critical_threshold
            ),
            largest_error_limit=5,
        )

        results.append(
            {
                "window_index":
                    len(results),

                "samples":
                    evaluation.total_samples,

                "mae":
                    evaluation.mae,

                "rmse":
                    evaluation.rmse,

                "bias":
                    evaluation.bias,

                "late_prediction_rate":
                    evaluation
                    .late_prediction_rate,

                "within_10_percent":
                    evaluation
                    .within_10_percent,

                "critical_recall":
                    evaluation
                    .critical_rul_detection_recall,
            }
        )

    return results


# =========================================================
# Evaluator service
# =========================================================

class RULEvaluator:

    def evaluate(
        self,
        samples: list[
            RULEvaluationSample
        ],
        *,
        critical_threshold: float = 24.0,
        absolute_tolerance: float = 12.0,
        largest_error_limit: int = 20,
    ) -> RULEvaluationResult:

        return evaluate_rul(
            samples,
            critical_threshold=(
                critical_threshold
            ),
            absolute_tolerance=(
                absolute_tolerance
            ),
            largest_error_limit=(
                largest_error_limit
            ),
        )

    def compare_models(
        self,
        model_samples: dict[
            str,
            list[
                RULEvaluationSample
            ],
        ],
        *,
        critical_threshold: float = 24.0,
    ) -> list[dict[str, Any]]:

        return compare_rul_models(
            model_samples,
            critical_threshold=(
                critical_threshold
            ),
        )

    def windows(
        self,
        samples: list[
            RULEvaluationSample
        ],
        *,
        window_size: int,
        critical_threshold: float = 24.0,
    ) -> list[dict[str, Any]]:

        return evaluate_rul_windows(
            samples,
            window_size=window_size,
            critical_threshold=(
                critical_threshold
            ),
        )


rul_evaluator = RULEvaluator()


__all__ = [
    "RULUnit",
    "RULRiskLevel",
    "RULErrorDirection",
    "RULPrediction",
    "RULGroundTruth",
    "RULEvaluationSample",
    "RULGroupMetrics",
    "RULEvaluationResult",
    "calculate_rul_error",
    "classify_error_direction",
    "calculate_mae",
    "calculate_mse",
    "calculate_rmse",
    "calculate_mape",
    "calculate_smape",
    "calculate_bias",
    "calculate_r2",
    "calculate_normalized_error",
    "calculate_direction_metrics",
    "calculate_asymmetric_score",
    "calculate_percentage_tolerance_accuracy",
    "calculate_absolute_tolerance_accuracy",
    "evaluate_critical_rul",
    "calculate_risk_level_accuracy",
    "calculate_interval_metrics",
    "calculate_group_metrics",
    "calculate_per_asset_metrics",
    "calculate_per_model_metrics",
    "get_largest_errors",
    "evaluate_rul",
    "compare_rul_models",
    "evaluate_rul_windows",
    "RULEvaluator",
    "rul_evaluator",
]