
"""
app/ai/ml/models/anomaly.py

Industrial Equipment Anomaly Detection
======================================

Python 3.11+

Dependencies:
    pip install numpy pandas scikit-learn joblib

Features:
- Isolation Forest anomaly detection
- Robust statistical anomaly detection
- Multivariate sensor monitoring
- Configurable contamination and thresholds
- Missing-value handling
- Feature scaling
- Model training and evaluation
- Batch and single-sample inference
- Anomaly scores
- Model persistence
- Feature schema validation
- Equipment-level metadata
- JSON-compatible results

Typical telemetry:
- vibration_mm_s
- temperature_c
- pressure_bar
- current_a
- voltage_v
- rpm

IMPORTANT:
    Anomaly detection identifies unusual
    observations relative to training data.

    It does not prove equipment failure,
    determine root cause or estimate RUL.

    Models must be validated on representative
    equipment and operating conditions.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class AnomalyModelError(Exception):
    """Base anomaly model exception."""


class AnomalyValidationError(AnomalyModelError):
    """Invalid input data or configuration."""


class AnomalyNotFittedError(AnomalyModelError):
    """Model inference attempted before fitting."""


# ============================================================
# ENUMS
# ============================================================

class AnomalyAlgorithm(str, Enum):
    ISOLATION_FOREST = "isolation_forest"
    ROBUST_STATISTICAL = "robust_statistical"


class AnomalySeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class AnomalyModelConfig:
    algorithm: AnomalyAlgorithm = (
        AnomalyAlgorithm.ISOLATION_FOREST
    )

    feature_columns: list[str] = field(
        default_factory=lambda: [
            "vibration_mm_s",
            "temperature_c",
            "pressure_bar",
            "current_a",
            "voltage_v",
            "rpm",
        ]
    )

    contamination: float = 0.03

    n_estimators: int = 200
    max_samples: int | float | str = "auto"

    random_state: int = 42

    min_training_samples: int = 30

    # Statistical model threshold:
    # robust z-score above threshold
    # is considered anomalous.
    statistical_z_threshold: float = 3.5

    # Severity bands for normalized
    # anomaly magnitude.
    low_severity_threshold: float = 0.0
    moderate_severity_threshold: float = 0.25
    high_severity_threshold: float = 0.50
    critical_severity_threshold: float = 0.75

    def __post_init__(self) -> None:
        if isinstance(self.algorithm, str):
            self.algorithm = AnomalyAlgorithm(
                self.algorithm
            )

        if not self.feature_columns:
            raise AnomalyValidationError(
                "feature_columns cannot be empty"
            )

        if (
            len(set(self.feature_columns))
            != len(self.feature_columns)
        ):
            raise AnomalyValidationError(
                "Duplicate feature names"
            )

        if not (
            0 < self.contamination <= 0.5
        ):
            raise AnomalyValidationError(
                "contamination must be in (0, 0.5]"
            )

        if self.n_estimators <= 0:
            raise AnomalyValidationError(
                "n_estimators must be positive"
            )

        if self.min_training_samples < 2:
            raise AnomalyValidationError(
                "min_training_samples must be >= 2"
            )

        if self.statistical_z_threshold <= 0:
            raise AnomalyValidationError(
                "statistical_z_threshold must be positive"
            )

        if not (
            0
            <= self.low_severity_threshold
            <= self.moderate_severity_threshold
            < self.high_severity_threshold
            < self.critical_severity_threshold
            <= 1
        ):
            raise AnomalyValidationError(
                "Invalid severity thresholds"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class AnomalyPrediction:
    prediction_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None

    is_anomaly: bool
    anomaly_score: float
    severity: str

    raw_model_score: float

    feature_values: dict[str, float | None]

    predicted_at: datetime

    requires_human_review: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "company_id": self.company_id,
            "asset_id": self.asset_id,
            "equipment_id": self.equipment_id,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "severity": self.severity,
            "raw_model_score": (
                self.raw_model_score
            ),
            "feature_values": dict(
                self.feature_values
            ),
            "predicted_at": (
                self.predicted_at.isoformat()
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AnomalyTrainingResult:
    model_id: str
    algorithm: str

    training_samples: int
    feature_count: int

    training_anomaly_count: int
    training_anomaly_rate: float

    trained_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "training_samples": (
                self.training_samples
            ),
            "feature_count": self.feature_count,
            "training_anomaly_count": (
                self.training_anomaly_count
            ),
            "training_anomaly_rate": (
                self.training_anomaly_rate
            ),
            "trained_at": (
                self.trained_at.isoformat()
            ),
        }


# ============================================================
# ROBUST STATISTICAL DETECTOR
# ============================================================

class RobustStatisticalDetector:
    """
    Median / MAD based multivariate
    anomaly screening.

    Uses maximum absolute robust z-score
    across features.

    This detects extreme feature deviations,
    but does not capture all correlations
    between sensor variables.
    """

    def __init__(
        self,
        threshold: float = 3.5,
    ) -> None:
        self.threshold = threshold

        self.medians_: np.ndarray | None = None
        self.mads_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
    ) -> "RobustStatisticalDetector":
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        self.medians_ = np.median(
            X,
            axis=0,
        )

        self.mads_ = np.median(
            np.abs(
                X - self.medians_
            ),
            axis=0,
        )

        self.mads_ = np.maximum(
            self.mads_,
            1e-6,
        )

        return self

    def anomaly_magnitude(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if (
            self.medians_ is None
            or self.mads_ is None
        ):
            raise AnomalyNotFittedError(
                "Statistical detector not fitted"
            )

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        robust_z = (
            0.6745
            * np.abs(
                X - self.medians_
            )
            / self.mads_
        )

        return np.max(
            robust_z,
            axis=1,
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        magnitude = self.anomaly_magnitude(
            X
        )

        return np.where(
            magnitude > self.threshold,
            -1,
            1,
        )

    def decision_function(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        magnitude = self.anomaly_magnitude(
            X
        )

        return (
            self.threshold - magnitude
        )


# ============================================================
# MAIN ANOMALY MODEL
# ============================================================

class IndustrialAnomalyModel:
    """
    End-to-end equipment anomaly model.

    Training:
        fit(dataframe)

    Inference:
        predict_one(sample, company_id=...)
        predict_batch(dataframe, company_id=...)

    Persistence:
        save(path)
        IndustrialAnomalyModel.load(path)

    NOTE:
        Only load trusted joblib files.
        joblib uses pickle internally.
    """

    def __init__(
        self,
        config: AnomalyModelConfig | None = None,
    ) -> None:
        self.config = (
            config or AnomalyModelConfig()
        )

        self.model_id = str(
            uuid.uuid4()
        )

        self._lock = threading.RLock()

        self._pipeline: Pipeline | None = None

        self._statistical_model: (
            RobustStatisticalDetector | None
        ) = None

        self._imputer: SimpleImputer | None = None

        self._is_fitted = False
        self._trained_at: datetime | None = None

        # Training-score scale used to
        # normalize anomaly magnitude.
        self._score_scale = 1.0

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _prepare_dataframe(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise AnomalyValidationError(
                "Expected pandas DataFrame"
            )

        missing = [
            column
            for column in self.config.feature_columns
            if column not in data.columns
        ]

        if missing:
            raise AnomalyValidationError(
                "Missing feature columns: "
                + ", ".join(missing)
            )

        selected = data[
            self.config.feature_columns
        ].copy()

        for column in selected.columns:
            selected[column] = (
                pd.to_numeric(
                    selected[column],
                    errors="raise",
                )
            )

        values = selected.to_numpy(
            dtype=np.float64
        )

        if np.isinf(values).any():
            raise AnomalyValidationError(
                "Infinite feature values are not allowed"
            )

        if selected.empty:
            raise AnomalyValidationError(
                "Input data cannot be empty"
            )

        return selected

    def _prepare_sample(
        self,
        sample: Mapping[str, Any],
    ) -> pd.DataFrame:
        if not isinstance(
            sample,
            Mapping,
        ):
            raise AnomalyValidationError(
                "Sample must be a mapping"
            )

        missing = [
            name
            for name in self.config.feature_columns
            if name not in sample
        ]

        if missing:
            raise AnomalyValidationError(
                "Missing sample features: "
                + ", ".join(missing)
            )

        return self._prepare_dataframe(
            pd.DataFrame(
                [
                    {
                        name: sample[name]
                        for name
                        in self.config.feature_columns
                    }
                ]
            )
        )

    @staticmethod
    def _validate_company(
        company_id: str,
    ) -> None:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise AnomalyValidationError(
                "company_id is required"
            )

    # ========================================================
    # TRAINING
    # ========================================================

    def fit(
        self,
        data: pd.DataFrame,
    ) -> AnomalyTrainingResult:
        frame = self._prepare_dataframe(
            data
        )

        if (
            len(frame)
            < self.config.min_training_samples
        ):
            raise AnomalyValidationError(
                "Insufficient training samples"
            )

        if frame.isna().all().any():
            raise AnomalyValidationError(
                "Training contains a feature "
                "with all missing values"
            )

        with self._lock:
            if (
                self.config.algorithm
                == AnomalyAlgorithm.ISOLATION_FOREST
            ):
                pipeline = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            RobustScaler(),
                        ),
                        (
                            "model",
                            IsolationForest(
                                n_estimators=(
                                    self.config.n_estimators
                                ),
                                max_samples=(
                                    self.config.max_samples
                                ),
                                contamination=(
                                    self.config.contamination
                                ),
                                random_state=(
                                    self.config.random_state
                                ),
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )

                pipeline.fit(
                    frame
                )

                raw_scores = (
                    pipeline.decision_function(
                        frame
                    )
                )

                labels = pipeline.predict(
                    frame
                )

                self._pipeline = pipeline
                self._statistical_model = None
                self._imputer = None

            else:
                imputer = SimpleImputer(
                    strategy="median"
                )

                X = imputer.fit_transform(
                    frame
                )

                statistical_model = (
                    RobustStatisticalDetector(
                        threshold=(
                            self.config
                            .statistical_z_threshold
                        )
                    )
                )

                statistical_model.fit(
                    X
                )

                raw_scores = (
                    statistical_model.decision_function(
                        X
                    )
                )

                labels = (
                    statistical_model.predict(
                        X
                    )
                )

                self._statistical_model = (
                    statistical_model
                )

                self._imputer = imputer
                self._pipeline = None

            # Positive magnitude means
            # more anomalous than the
            # model's decision boundary.
            magnitudes = np.maximum(
                0.0,
                -raw_scores,
            )

            positive = magnitudes[
                magnitudes > 0
            ]

            if len(positive):
                self._score_scale = max(
                    float(
                        np.percentile(
                            positive,
                            95,
                        )
                    ),
                    1e-6,
                )
            else:
                self._score_scale = 1.0

            self._is_fitted = True

            self._trained_at = datetime.now(
                timezone.utc
            )

            anomaly_count = int(
                np.sum(labels == -1)
            )

            return AnomalyTrainingResult(
                model_id=self.model_id,
                algorithm=(
                    self.config.algorithm.value
                ),
                training_samples=len(frame),
                feature_count=(
                    len(
                        self.config.feature_columns
                    )
                ),
                training_anomaly_count=(
                    anomaly_count
                ),
                training_anomaly_rate=round(
                    anomaly_count
                    / len(frame),
                    4,
                ),
                trained_at=self._trained_at,
            )

    # ========================================================
    # RAW INFERENCE
    # ========================================================

    def _predict_raw(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise AnomalyNotFittedError(
                "Train or load model before inference"
            )

        if (
            self.config.algorithm
            == AnomalyAlgorithm.ISOLATION_FOREST
        ):
            if self._pipeline is None:
                raise AnomalyNotFittedError(
                    "Isolation Forest pipeline unavailable"
                )

            labels = self._pipeline.predict(
                frame
            )

            scores = (
                self._pipeline.decision_function(
                    frame
                )
            )

        else:
            if (
                self._statistical_model is None
                or self._imputer is None
            ):
                raise AnomalyNotFittedError(
                    "Statistical model unavailable"
                )

            X = self._imputer.transform(
                frame
            )

            labels = (
                self._statistical_model.predict(
                    X
                )
            )

            scores = (
                self._statistical_model.decision_function(
                    X
                )
            )

        return (
            np.asarray(labels),
            np.asarray(scores),
        )

    # ========================================================
    # NORMALIZED ANOMALY MAGNITUDE
    # ========================================================

    def _normalize_score(
        self,
        raw_score: float,
    ) -> float:
        magnitude = max(
            0.0,
            -raw_score,
        )

        # Saturating transform to [0,1).
        # This is NOT a calibrated probability.
        normalized = (
            magnitude
            / (
                magnitude
                + self._score_scale
            )
        )

        return float(
            max(
                0.0,
                min(1.0, normalized),
            )
        )

    # ========================================================
    # SEVERITY
    # ========================================================

    def _severity(
        self,
        *,
        is_anomaly: bool,
        anomaly_score: float,
    ) -> AnomalySeverity:
        if not is_anomaly:
            return AnomalySeverity.NORMAL

        if (
            anomaly_score
            >= self.config
            .critical_severity_threshold
        ):
            return AnomalySeverity.CRITICAL

        if (
            anomaly_score
            >= self.config.high_severity_threshold
        ):
            return AnomalySeverity.HIGH

        if (
            anomaly_score
            >= self.config.moderate_severity_threshold
        ):
            return AnomalySeverity.MODERATE

        return AnomalySeverity.LOW

    # ========================================================
    # SINGLE PREDICTION
    # ========================================================

    def predict_one(
        self,
        sample: Mapping[str, Any],
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AnomalyPrediction:
        self._validate_company(
            company_id
        )

        frame = self._prepare_sample(
            sample
        )

        with self._lock:
            labels, raw_scores = (
                self._predict_raw(
                    frame
                )
            )

            label = int(labels[0])
            raw_score = float(
                raw_scores[0]
            )

            normalized = (
                self._normalize_score(
                    raw_score
                )
            )

        is_anomaly = (
            label == -1
        )

        severity = self._severity(
            is_anomaly=is_anomaly,
            anomaly_score=normalized,
        )

        feature_values = {}

        for feature in self.config.feature_columns:
            value = frame.iloc[0][feature]

            feature_values[feature] = (
                None
                if pd.isna(value)
                else float(value)
            )

        return AnomalyPrediction(
            prediction_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            is_anomaly=is_anomaly,
            anomaly_score=round(
                normalized,
                6,
            ),
            severity=severity.value,
            raw_model_score=round(
                raw_score,
                6,
            ),
            feature_values=feature_values,
            predicted_at=datetime.now(
                timezone.utc
            ),
            requires_human_review=(
                is_anomaly
            ),
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(
        self,
        data: pd.DataFrame,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> list[AnomalyPrediction]:
        self._validate_company(
            company_id
        )

        frame = self._prepare_dataframe(
            data
        )

        with self._lock:
            labels, raw_scores = (
                self._predict_raw(
                    frame
                )
            )

            normalized_scores = [
                self._normalize_score(
                    float(raw)
                )
                for raw in raw_scores
            ]

        predictions: list[
            AnomalyPrediction
        ] = []

        now = datetime.now(
            timezone.utc
        )

        for index in range(len(frame)):
            is_anomaly = (
                int(labels[index]) == -1
            )

            normalized = (
                normalized_scores[index]
            )

            severity = self._severity(
                is_anomaly=is_anomaly,
                anomaly_score=normalized,
            )

            row = frame.iloc[index]

            feature_values = {
                feature: (
                    None
                    if pd.isna(row[feature])
                    else float(row[feature])
                )
                for feature
                in self.config.feature_columns
            }

            predictions.append(
                AnomalyPrediction(
                    prediction_id=str(
                        uuid.uuid4()
                    ),
                    company_id=company_id,
                    asset_id=asset_id,
                    equipment_id=equipment_id,
                    is_anomaly=is_anomaly,
                    anomaly_score=round(
                        normalized,
                        6,
                    ),
                    severity=severity.value,
                    raw_model_score=round(
                        float(
                            raw_scores[index]
                        ),
                        6,
                    ),
                    feature_values=feature_values,
                    predicted_at=now,
                    requires_human_review=(
                        is_anomaly
                    ),
                )
            )

        return predictions

    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(
        self,
        data: pd.DataFrame,
        y_true: Sequence[int],
    ) -> dict[str, Any]:
        """
        Expected ground truth:
            0 = normal
            1 = anomaly
        """

        frame = self._prepare_dataframe(
            data
        )

        if len(y_true) != len(frame):
            raise AnomalyValidationError(
                "Ground-truth length mismatch"
            )

        y_true_array = np.asarray(
            y_true,
            dtype=np.int64,
        )

        if not np.isin(
            y_true_array,
            [0, 1],
        ).all():
            raise AnomalyValidationError(
                "Ground truth must contain 0 or 1"
            )

        with self._lock:
            labels, _ = self._predict_raw(
                frame
            )

        y_pred = (
            labels == -1
        ).astype(np.int64)

        matrix = confusion_matrix(
            y_true_array,
            y_pred,
            labels=[0, 1],
        )

        return {
            "precision": float(
                precision_score(
                    y_true_array,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true_array,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_true_array,
                    y_pred,
                    zero_division=0,
                )
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
            "samples": len(frame),
        }

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        if not self._is_fitted:
            raise AnomalyNotFittedError(
                "Cannot save unfitted model"
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "model_id": self.model_id,
                "config": asdict(self.config),
                "pipeline": self._pipeline,
                "statistical_model": (
                    self._statistical_model
                ),
                "imputer": self._imputer,
                "score_scale": (
                    self._score_scale
                ),
                "trained_at": (
                    self._trained_at.isoformat()
                    if self._trained_at
                    else None
                ),
            }

            payload["config"]["algorithm"] = (
                self.config.algorithm.value
            )

            joblib.dump(
                payload,
                path,
            )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "IndustrialAnomalyModel":
        """
        WARNING:
            Load only trusted model files.
            joblib/pickle deserialization can
            execute arbitrary code.
        """

        path = Path(path)

        if not path.is_file():
            raise AnomalyModelError(
                "Model file not found"
            )

        payload = joblib.load(
            path
        )

        config = AnomalyModelConfig(
            **payload["config"]
        )

        model = cls(
            config
        )

        model.model_id = payload["model_id"]

        model._pipeline = payload["pipeline"]

        model._statistical_model = (
            payload["statistical_model"]
        )

        model._imputer = payload["imputer"]

        model._score_scale = float(
            payload["score_scale"]
        )

        trained_at = payload.get(
            "trained_at"
        )

        model._trained_at = (
            datetime.fromisoformat(
                trained_at
            )
            if trained_at
            else None
        )

        model._is_fitted = True

        return model

    # ========================================================
    # MODEL INFO
    # ========================================================

    def get_model_info(
        self,
    ) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "algorithm": (
                self.config.algorithm.value
            ),
            "feature_columns": list(
                self.config.feature_columns
            ),
            "is_fitted": self._is_fitted,
            "trained_at": (
                self._trained_at.isoformat()
                if self._trained_at
                else None
            ),
        }


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    rng = np.random.default_rng(
        42
    )

    training_data = pd.DataFrame({
        "vibration_mm_s": rng.normal(
            3.0, 0.5, 500
        ),
        "temperature_c": rng.normal(
            55.0, 4.0, 500
        ),
        "pressure_bar": rng.normal(
            6.0, 0.4, 500
        ),
        "current_a": rng.normal(
            12.0, 1.0, 500
        ),
        "voltage_v": rng.normal(
            230.0, 3.0, 500
        ),
        "rpm": rng.normal(
            1450.0, 40.0, 500
        ),
    })

    model = IndustrialAnomalyModel(
        AnomalyModelConfig(
            algorithm=(
                AnomalyAlgorithm.ISOLATION_FOREST
            ),
            contamination=0.03,
        )
    )

    training_result = model.fit(
        training_data
    )

    print(
        json.dumps(
            training_result.to_dict(),
            indent=2,
        )
    )

    sample = {
        "vibration_mm_s": 11.5,
        "temperature_c": 105.0,
        "pressure_bar": 9.0,
        "current_a": 23.0,
        "voltage_v": 215.0,
        "rpm": 1750.0,
    }

    prediction = model.predict_one(
        sample,
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="PUMP-101",
    )

    print(
        json.dumps(
            prediction.to_dict(),
            indent=2,
        )
    )

    model.save(
        "models/anomaly/isolation_forest.joblib"
    )

    restored = IndustrialAnomalyModel.load(
        "models/anomaly/isolation_forest.joblib"
    )

    print(
        restored.get_model_info()
    )


if __name__ == "__main__":
    example()
