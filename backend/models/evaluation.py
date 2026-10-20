"""
app/ai/ml/models/evaluation.py

Industrial AI Model Evaluation
==============================

Python 3.11+
Pydantic v2, NumPy, scikit-learn

Install:
    pip install "pydantic>=2,<3" numpy scikit-learn

Features:
- Classification evaluation
- Anomaly detection evaluation
- Failure prediction evaluation
- Remaining useful life (RUL) evaluation
- Time-series forecasting evaluation
- Regression metrics
- Confusion matrix
- Precision, recall, F1
- Accuracy and balanced accuracy
- ROC-AUC and PR-AUC
- MAE, MSE, RMSE, R2
- MAPE and sMAPE
- Model benchmarking
- Configurable acceptance thresholds
- Model evaluation reports
- Multi-tenant evaluation metadata
- JSON-compatible results

IMPORTANT:
    Evaluation metrics alone do not prove
    operational or safety readiness.

    Use independent holdout datasets,
    time-aware validation, representative
    equipment and operating conditions.

    Thresholds must be approved for the
    specific industrial application.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelEvaluationError(Exception):
    """Base evaluation exception."""


class EvaluationValidationError(ModelEvaluationError):
    """Invalid evaluation inputs."""


# ============================================================
# ENUMS
# ============================================================

class EvaluationTask(str, Enum):
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_PREDICTION = "rul_prediction"
    REGRESSION = "regression"
    FORECASTING = "forecasting"


class EvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = (
        "insufficient_evidence"
    )


class EvaluationDatasetType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BACKTEST = "backtest"
    PRODUCTION = "production"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


# ============================================================
# BASE MODEL
# ============================================================

class EvaluationBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# MODEL REFERENCE
# ============================================================

class EvaluationModelReference(EvaluationBaseModel):
    model_id: str = Field(
        min_length=1,
        max_length=200,
    )

    model_name: str = Field(
        min_length=1,
        max_length=200,
    )

    model_version: str = Field(
        min_length=1,
        max_length=100,
    )

    algorithm: str | None = None
    training_run_id: str | None = None

    feature_schema_version: str | None = None


# ============================================================
# DATASET REFERENCE
# ============================================================

class EvaluationDatasetReference(EvaluationBaseModel):
    dataset_id: str = Field(
        min_length=1,
        max_length=200,
    )

    dataset_version: str | None = None

    dataset_type: EvaluationDatasetType

    sample_count: int = Field(
        ge=1,
    )

    asset_type: str | None = None

    start_time: datetime | None = None
    end_time: datetime | None = None

    @model_validator(mode="after")
    def validate_dates(
        self,
    ) -> "EvaluationDatasetReference":
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError(
                "Dataset end_time precedes start_time"
            )

        for timestamp in (
            self.start_time,
            self.end_time,
        ):
            if (
                timestamp is not None
                and timestamp.tzinfo is None
            ):
                raise ValueError(
                    "Dataset timestamps must "
                    "be timezone-aware"
                )

        return self


# ============================================================
# METRIC RESULT
# ============================================================

class EvaluationMetric(EvaluationBaseModel):
    metric_name: str = Field(
        min_length=1,
        max_length=100,
    )

    value: float | None = None

    direction: MetricDirection

    threshold: float | None = None

    passed: bool | None = None

    description: str | None = None

    @model_validator(mode="after")
    def validate_metric(
        self,
    ) -> "EvaluationMetric":
        for number in (
            self.value,
            self.threshold,
        ):
            if (
                number is not None
                and not math.isfinite(number)
            ):
                raise ValueError(
                    "Metric values must be finite"
                )

        return self


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

class ClassificationMetrics(EvaluationBaseModel):
    accuracy: float = Field(
        ge=0,
        le=1,
    )

    balanced_accuracy: float = Field(
        ge=0,
        le=1,
    )

    precision: float = Field(
        ge=0,
        le=1,
    )

    recall: float = Field(
        ge=0,
        le=1,
    )

    f1_score: float = Field(
        ge=0,
        le=1,
    )

    roc_auc: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    average_precision: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_positive: int = Field(ge=0)

    sample_count: int = Field(
        ge=1,
    )

    positive_support: int = Field(
        ge=0,
    )

    negative_support: int = Field(
        ge=0,
    )


# ============================================================
# REGRESSION METRICS
# ============================================================

class RegressionMetrics(EvaluationBaseModel):
    mae: float = Field(ge=0)
    mse: float = Field(ge=0)
    rmse: float = Field(ge=0)

    r2: float | None = None

    mape_percent: float | None = Field(
        default=None,
        ge=0,
    )

    smape_percent: float | None = Field(
        default=None,
        ge=0,
        le=200,
    )

    sample_count: int = Field(
        ge=1,
    )


# ============================================================
# ACCEPTANCE RULE
# ============================================================

class EvaluationAcceptanceRule(EvaluationBaseModel):
    metric_name: str = Field(
        min_length=1,
        max_length=100,
    )

    threshold: float

    direction: MetricDirection

    required: bool = True

    @model_validator(mode="after")
    def validate_threshold(
        self,
    ) -> "EvaluationAcceptanceRule":
        if not math.isfinite(self.threshold):
            raise ValueError(
                "Threshold must be finite"
            )

        return self


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

class ModelEvaluationConfig(EvaluationBaseModel):
    task: EvaluationTask

    positive_label: int = 1

    # Used only for classification
    # when continuous scores are supplied.
    decision_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )

    # Avoid undefined MAPE for targets
    # equal to or near zero.
    mape_epsilon: float = Field(
        default=1e-8,
        gt=0,
    )

    acceptance_rules: list[
        EvaluationAcceptanceRule
    ] = Field(
        default_factory=list
    )


# ============================================================
# EVALUATION REPORT
# ============================================================

class ModelEvaluationReport(EvaluationBaseModel):
    evaluation_id: UUID = Field(
        default_factory=uuid4
    )

    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    task: EvaluationTask

    model: EvaluationModelReference

    dataset: EvaluationDatasetReference

    classification_metrics: (
        ClassificationMetrics | None
    ) = None

    regression_metrics: (
        RegressionMetrics | None
    ) = None

    metrics: list[EvaluationMetric] = Field(
        default_factory=list
    )

    status: EvaluationStatus = (
        EvaluationStatus.INSUFFICIENT_EVIDENCE
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )


# ============================================================
# INPUT VALIDATION
# ============================================================

class EvaluationInputValidator:

    @staticmethod
    def numeric_vectors(
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            actual = np.asarray(
                y_true,
                dtype=np.float64,
            )

            predicted = np.asarray(
                y_pred,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationValidationError(
                "Inputs must be numeric"
            ) from exc

        if (
            actual.ndim != 1
            or predicted.ndim != 1
        ):
            raise EvaluationValidationError(
                "Inputs must be one-dimensional"
            )

        if (
            len(actual) == 0
            or len(actual) != len(predicted)
        ):
            raise EvaluationValidationError(
                "Input arrays must be non-empty "
                "and have matching lengths"
            )

        if (
            not np.isfinite(actual).all()
            or not np.isfinite(predicted).all()
        ):
            raise EvaluationValidationError(
                "Inputs must contain finite values"
            )

        return actual, predicted

    @staticmethod
    def binary_labels(
        values: np.ndarray,
        *,
        name: str,
    ) -> None:
        if not np.isin(
            values,
            [0, 1],
        ).all():
            raise EvaluationValidationError(
                f"{name} must contain only 0 and 1"
            )


# ============================================================
# CLASSIFICATION EVALUATOR
# ============================================================

class ClassificationEvaluator:

    @staticmethod
    def evaluate(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        *,
        y_score: Sequence[float] | None = None,
    ) -> ClassificationMetrics:
        actual, predicted = (
            EvaluationInputValidator.numeric_vectors(
                y_true,
                y_pred,
            )
        )

        EvaluationInputValidator.binary_labels(
            actual,
            name="y_true",
        )

        EvaluationInputValidator.binary_labels(
            predicted,
            name="y_pred",
        )

        actual = actual.astype(np.int64)
        predicted = predicted.astype(np.int64)

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        )

        tn, fp, fn, tp = (
            int(value)
            for value in matrix.ravel()
        )

        roc_auc = None
        average_precision = None

        if y_score is not None:
            _, scores = (
                EvaluationInputValidator.numeric_vectors(
                    actual,
                    y_score,
                )
            )

            if (
                (scores < 0).any()
                or (scores > 1).any()
            ):
                raise EvaluationValidationError(
                    "y_score must be in [0, 1]"
                )

            if len(np.unique(actual)) == 2:
                roc_auc = float(
                    roc_auc_score(
                        actual,
                        scores,
                    )
                )

                average_precision = float(
                    average_precision_score(
                        actual,
                        scores,
                    )
                )

        return ClassificationMetrics(
            accuracy=float(
                accuracy_score(
                    actual,
                    predicted,
                )
            ),
            balanced_accuracy=float(
                balanced_accuracy_score(
                    actual,
                    predicted,
                )
            ),
            precision=float(
                precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            f1_score=float(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            roc_auc=roc_auc,
            average_precision=average_precision,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            true_positive=tp,
            sample_count=len(actual),
            positive_support=int(
                np.sum(actual == 1)
            ),
            negative_support=int(
                np.sum(actual == 0)
            ),
        )


# ============================================================
# REGRESSION / RUL EVALUATOR
# ============================================================

class RegressionEvaluator:

    @staticmethod
    def evaluate(
        y_true: Sequence[float],
        y_pred: Sequence[float],
        *,
        epsilon: float = 1e-8,
    ) -> RegressionMetrics:
        actual, predicted = (
            EvaluationInputValidator.numeric_vectors(
                y_true,
                y_pred,
            )
        )

        errors = (
            actual - predicted
        )

        mae = float(
            mean_absolute_error(
                actual,
                predicted,
            )
        )

        mse = float(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        rmse = math.sqrt(mse)

        r2 = None

        if (
            len(actual) >= 2
            and not np.allclose(
                actual,
                actual[0],
            )
        ):
            value = float(
                r2_score(
                    actual,
                    predicted,
                )
            )

            if math.isfinite(value):
                r2 = value

        nonzero_mask = (
            np.abs(actual) > epsilon
        )

        mape = None

        if np.any(nonzero_mask):
            mape = float(
                np.mean(
                    np.abs(
                        errors[nonzero_mask]
                        / actual[nonzero_mask]
                    )
                )
                * 100.0
            )

        denominator = (
            np.abs(actual)
            + np.abs(predicted)
        )

        valid_mask = (
            denominator > epsilon
        )

        smape = 0.0

        if np.any(valid_mask):
            smape = float(
                np.mean(
                    200.0
                    * np.abs(
                        errors[valid_mask]
                    )
                    / denominator[valid_mask]
                )
            )

        return RegressionMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            mape_percent=mape,
            smape_percent=smape,
            sample_count=len(actual),
        )


# ============================================================
# ACCEPTANCE EVALUATOR
# ============================================================

class ModelAcceptanceEvaluator:

    @staticmethod
    def evaluate(
        metric_values: Mapping[
            str,
            float | None,
        ],
        rules: Sequence[
            EvaluationAcceptanceRule
        ],
    ) -> tuple[
        EvaluationStatus,
        list[EvaluationMetric],
    ]:
        results: list[EvaluationMetric] = []

        if not rules:
            return (
                EvaluationStatus.INSUFFICIENT_EVIDENCE,
                results,
            )

        required_failed = False
        required_missing = False

        for rule in rules:
            value = metric_values.get(
                rule.metric_name
            )

            passed = None

            if value is not None:
                if (
                    rule.direction
                    == MetricDirection.HIGHER_IS_BETTER.value
                ):
                    passed = (
                        value >= rule.threshold
                    )
                else:
                    passed = (
                        value <= rule.threshold
                    )

            if rule.required:
                if passed is False:
                    required_failed = True

                elif passed is None:
                    required_missing = True

            results.append(
                EvaluationMetric(
                    metric_name=rule.metric_name,
                    value=value,
                    threshold=rule.threshold,
                    direction=rule.direction,
                    passed=passed,
                )
            )

        if required_failed:
            status = EvaluationStatus.FAILED

        elif required_missing:
            status = (
                EvaluationStatus.INSUFFICIENT_EVIDENCE
            )

        else:
            status = EvaluationStatus.PASSED

        return status, results


# ============================================================
# MAIN MODEL EVALUATOR
# ============================================================

class IndustrialModelEvaluator:
    """
    Evaluate industrial ML predictions.

    Supports:
    - Classification
    - Anomaly detection
    - Failure prediction
    - RUL prediction
    - Regression
    - Forecasting

    For anomaly evaluation:
        y_true: 0=normal, 1=anomaly
        y_pred: 0=normal, 1=anomaly

    For failure prediction:
        y_true: 0=no failure, 1=failure

    For regression:
        y_true and y_pred are numeric targets.

    Classification score arrays are expected
    to contain probabilities or normalized
    ranking scores between 0 and 1.
    """

    CLASSIFICATION_TASKS = {
        EvaluationTask.CLASSIFICATION,
        EvaluationTask.ANOMALY_DETECTION,
        EvaluationTask.FAILURE_PREDICTION,
    }

    REGRESSION_TASKS = {
        EvaluationTask.REGRESSION,
        EvaluationTask.RUL_PREDICTION,
        EvaluationTask.FORECASTING,
    }

    def __init__(
        self,
        config: ModelEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        company_id: str,
        model: EvaluationModelReference,
        dataset: EvaluationDatasetReference,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        y_score: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelEvaluationReport:
        if not company_id.strip():
            raise EvaluationValidationError(
                "company_id is required"
            )

        if len(y_true) != dataset.sample_count:
            raise EvaluationValidationError(
                "Dataset sample count mismatch"
            )

        task = EvaluationTask(
            self.config.task
        )

        warnings: list[str] = []

        classification_metrics = None
        regression_metrics = None

        if task in self.CLASSIFICATION_TASKS:
            classification_metrics = (
                ClassificationEvaluator.evaluate(
                    y_true,
                    y_pred,
                    y_score=y_score,
                )
            )

            metric_values = {
                "accuracy": (
                    classification_metrics.accuracy
                ),
                "balanced_accuracy": (
                    classification_metrics.balanced_accuracy
                ),
                "precision": (
                    classification_metrics.precision
                ),
                "recall": (
                    classification_metrics.recall
                ),
                "f1_score": (
                    classification_metrics.f1_score
                ),
                "roc_auc": (
                    classification_metrics.roc_auc
                ),
                "average_precision": (
                    classification_metrics.average_precision
                ),
            }

            if (
                classification_metrics.positive_support == 0
                or classification_metrics.negative_support == 0
            ):
                warnings.append(
                    "Evaluation contains only one "
                    "ground-truth class. Some "
                    "classification metrics are "
                    "undefined or less informative."
                )

            if y_score is None:
                warnings.append(
                    "No continuous prediction scores "
                    "were supplied; ROC-AUC and "
                    "average precision are unavailable."
                )

        elif task in self.REGRESSION_TASKS:
            regression_metrics = (
                RegressionEvaluator.evaluate(
                    y_true,
                    y_pred,
                    epsilon=(
                        self.config.mape_epsilon
                    ),
                )
            )

            metric_values = {
                "mae": regression_metrics.mae,
                "mse": regression_metrics.mse,
                "rmse": regression_metrics.rmse,
                "r2": regression_metrics.r2,
                "mape_percent": (
                    regression_metrics.mape_percent
                ),
                "smape_percent": (
                    regression_metrics.smape_percent
                ),
            }

            if (
                regression_metrics.mape_percent
                is None
            ):
                warnings.append(
                    "MAPE is unavailable because "
                    "all actual targets are zero "
                    "or near zero."
                )

        else:
            raise EvaluationValidationError(
                f"Unsupported task: {task}"
            )

        status, acceptance_metrics = (
            ModelAcceptanceEvaluator.evaluate(
                metric_values,
                self.config.acceptance_rules,
            )
        )

        if (
            dataset.dataset_type
            == EvaluationDatasetType.TRAIN.value
        ):
            warnings.append(
                "Training-set evaluation cannot "
                "establish generalization."
            )

        if not self.config.acceptance_rules:
            warnings.append(
                "No acceptance rules configured; "
                "readiness status is "
                "insufficient_evidence."
            )

        return ModelEvaluationReport(
            company_id=company_id,
            task=task,
            model=model,
            dataset=dataset,
            classification_metrics=(
                classification_metrics
            ),
            regression_metrics=(
                regression_metrics
            ),
            metrics=acceptance_metrics,
            status=status,
            warnings=warnings,
            metadata=dict(metadata or {}),
        )


# ============================================================
# MODEL BENCHMARKING
# ============================================================

class ModelBenchmarkResult(EvaluationBaseModel):
    benchmark_id: UUID = Field(
        default_factory=uuid4
    )

    task: EvaluationTask

    metric_name: str

    direction: MetricDirection

    ranked_model_ids: list[str]

    metric_values: dict[str, float]

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


class IndustrialModelBenchmark:

    @staticmethod
    def compare(
        reports: Sequence[
            ModelEvaluationReport
        ],
        *,
        metric_name: str,
        direction: MetricDirection,
    ) -> ModelBenchmarkResult:
        if not reports:
            raise EvaluationValidationError(
                "At least one report is required"
            )

        task = reports[0].task

        if any(
            report.task != task
            for report in reports
        ):
            raise EvaluationValidationError(
                "Cannot compare different tasks"
            )

        dataset_identity = (
            reports[0].dataset.dataset_id,
            reports[0].dataset.dataset_version,
            reports[0].dataset.dataset_type,
        )

        if any(
            (
                report.dataset.dataset_id,
                report.dataset.dataset_version,
                report.dataset.dataset_type,
            ) != dataset_identity
            for report in reports
        ):
            raise EvaluationValidationError(
                "Benchmark reports must use "
                "the same dataset and version"
            )

        values: dict[str, float] = {}

        for report in reports:
            metrics = {}

            if report.classification_metrics:
                metrics.update(
                    report.classification_metrics.model_dump()
                )

            if report.regression_metrics:
                metrics.update(
                    report.regression_metrics.model_dump()
                )

            value = metrics.get(
                metric_name
            )

            if value is None:
                raise EvaluationValidationError(
                    f"Metric unavailable: {metric_name}"
                )

            model_key = (
                f"{report.model.model_id}:"
                f"{report.model.model_version}"
            )

            if model_key in values:
                raise EvaluationValidationError(
                    "Duplicate model/version in benchmark"
                )

            values[model_key] = float(value)

        descending = (
            direction
            == MetricDirection.HIGHER_IS_BETTER.value
            or direction
            == MetricDirection.HIGHER_IS_BETTER
        )

        ranked = sorted(
            values,
            key=values.get,
            reverse=descending,
        )

        return ModelBenchmarkResult(
            task=task,
            metric_name=metric_name,
            direction=direction,
            ranked_model_ids=ranked,
            metric_values=values,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    config = ModelEvaluationConfig(
        task=EvaluationTask.ANOMALY_DETECTION,
        acceptance_rules=[
            EvaluationAcceptanceRule(
                metric_name="recall",
                threshold=0.80,
                direction=(
                    MetricDirection.HIGHER_IS_BETTER
                ),
            ),
            EvaluationAcceptanceRule(
                metric_name="precision",
                threshold=0.70,
                direction=(
                    MetricDirection.HIGHER_IS_BETTER
                ),
            ),
        ],
    )

    evaluator = IndustrialModelEvaluator(
        config
    )

    y_true = [
        0, 0, 0, 1, 0,
        1, 0, 0, 1, 0,
    ]

    y_pred = [
        0, 0, 0, 1, 0,
        1, 0, 1, 1, 0,
    ]

    y_score = [
        0.05, 0.10, 0.15, 0.90, 0.08,
        0.85, 0.12, 0.72, 0.95, 0.06,
    ]

    report = evaluator.evaluate(
        company_id="company-001",
        model=EvaluationModelReference(
            model_id="anomaly-model-001",
            model_name="Industrial Isolation Forest",
            model_version="1.0.0",
            algorithm="isolation_forest",
        ),
        dataset=EvaluationDatasetReference(
            dataset_id="industrial-test-001",
            dataset_version="1.0",
            dataset_type=EvaluationDatasetType.TEST,
            sample_count=len(y_true),
        ),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()