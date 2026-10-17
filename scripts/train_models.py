"""
scripts/train_models.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Machine Learning Training Pipeline
==================================

Python 3.11+

Install:
    pip install numpy pandas scikit-learn joblib

Optional:
    pip install mlflow

Tasks:
    anomaly       - Sensor anomaly classification
    failure       - Failure-within-horizon classification
    rul           - Remaining Useful Life regression
    all           - Train all available tasks

Models:
    RandomForestClassifier
    HistGradientBoostingClassifier
    IsolationForest
    RandomForestRegressor
    HistGradientBoostingRegressor

Features:
    - Synthetic dataset loading
    - Input schema validation
    - Numeric feature engineering
    - Missing-value imputation
    - Train/validation/test splitting
    - Asset-grouped splitting
    - Time-ordered splitting
    - Model selection
    - Optional hyperparameter search
    - Classification evaluation
    - Regression evaluation
    - Confusion matrices
    - Feature importance
    - Model artifact persistence
    - Optional MLflow tracking
    - Reproducible training
    - JSON reports
    - CLI

Important:
    Synthetic labels are not production-validated.
    Simulator latent variables must not be used
    as ordinary predictive features.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline


LOGGER = logging.getLogger("industrial_model_training")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str = "data/synthetic"

    output_dir: str = "artifacts/models"

    task: Literal[
        "anomaly",
        "failure",
        "rul",
        "all",
    ] = "all"

    seed: int = 42

    test_size: float = 0.20
    validation_size: float = 0.20

    split_strategy: Literal[
        "asset",
        "time",
    ] = "asset"

    tune: bool = False
    tuning_iterations: int = 10

    enable_mlflow: bool = False

    mlflow_experiment: str = (
        "industrial_predictive_maintenance"
    )

    max_rows: int | None = None

    min_training_rows: int = 30


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

SENSOR_FEATURES = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "current_a",
    "voltage_v",
    "rpm",
    "power_kw",
    "flow_rate_l_min",
    "humidity_percent",
    "acoustic_db",
    "operating_hours",
]

# Deliberately excluded:
#
# health_score
# degradation
# simulated_failure_probability
# failure_event
# failure_id
# next_failure_at
# rul_hours
# failure_within_horizon
# anomaly_label
# anomaly_type
# alert_count
#
# These may leak simulator state, future outcomes
# or downstream detection decisions.


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):
        value = float(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            json_safe(data),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_dataset(
    path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        nrows=max_rows,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    if "timestamp" in dataframe.columns:
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
            errors="coerce",
        )

    return dataframe


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class IndustrialFeatureEngineer:
    @staticmethod
    def transform(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        data = dataframe.copy()

        for column in SENSOR_FEATURES:
            if column not in data.columns:
                data[column] = np.nan

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        if "timestamp" in data.columns:
            timestamps = pd.to_datetime(
                data["timestamp"],
                utc=True,
                errors="coerce",
            )

            data["hour"] = timestamps.dt.hour
            data["day_of_week"] = (
                timestamps.dt.dayofweek
            )
            data["month"] = timestamps.dt.month
        else:
            data["hour"] = np.nan
            data["day_of_week"] = np.nan
            data["month"] = np.nan

        # Derived features use only measurements
        # available at the current observation.
        data["temperature_vibration"] = (
            data["temperature_c"]
            * data["vibration_mm_s"]
        )

        data["current_voltage"] = (
            data["current_a"]
            * data["voltage_v"]
        )

        data["power_per_current"] = (
            data["power_kw"]
            / data["current_a"].replace(
                0,
                np.nan,
            )
        )

        data["vibration_per_rpm"] = (
            data["vibration_mm_s"]
            / data["rpm"].replace(
                0,
                np.nan,
            )
        )

        data["pressure_flow_ratio"] = (
            data["pressure_bar"]
            / data["flow_rate_l_min"].replace(
                0,
                np.nan,
            )
        )

        features = SENSOR_FEATURES + [
            "hour",
            "day_of_week",
            "month",
            "temperature_vibration",
            "current_voltage",
            "power_per_current",
            "vibration_per_rpm",
            "pressure_flow_ratio",
        ]

        result = data[
            features
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return result.astype(float)


# ============================================================
# DATA SPLITTING
# ============================================================

class DatasetSplitter:
    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        self.config = config

    def split(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Returns positional train, validation and
        test indices.

        Asset strategy:
            No asset appears in multiple splits.

        Time strategy:
            Chronological train/validation/test.

        Time strategy alone does not eliminate
        leakage from overlapping prediction horizons.
        """

        if self.config.split_strategy == "asset":
            return self._asset_split(
                dataframe
            )

        return self._time_split(
            dataframe
        )

    def _asset_split(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if "asset_id" not in dataframe.columns:
            raise ValueError(
                "asset_id required for asset splitting"
            )

        groups = dataframe[
            "asset_id"
        ].astype(str).to_numpy()

        if len(np.unique(groups)) < 3:
            raise ValueError(
                "At least three assets are required "
                "for train/validation/test splitting"
            )

        indices = np.arange(
            len(dataframe)
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.test_size,
            random_state=self.config.seed,
        )

        train_val_idx, test_idx = next(
            splitter.split(
                indices,
                groups=groups,
            )
        )

        relative_validation_size = (
            self.config.validation_size
            / (1 - self.config.test_size)
        )

        second_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_validation_size,
            random_state=self.config.seed + 1,
        )

        train_relative, val_relative = next(
            second_splitter.split(
                train_val_idx,
                groups=groups[train_val_idx],
            )
        )

        train_idx = train_val_idx[
            train_relative
        ]

        val_idx = train_val_idx[
            val_relative
        ]

        return (
            train_idx,
            val_idx,
            test_idx,
        )

    def _time_split(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if "timestamp" not in dataframe.columns:
            raise ValueError(
                "timestamp required for time splitting"
            )

        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                "Invalid timestamps in dataset"
            )

        indices = np.argsort(
            timestamps.to_numpy(),
            kind="stable",
        )

        total = len(indices)

        test_start = int(
            total * (1 - self.config.test_size)
        )

        val_start = int(
            total * (
                1
                - self.config.test_size
                - self.config.validation_size
            )
        )

        return (
            indices[:val_start],
            indices[val_start:test_start],
            indices[test_start:],
        )


# ============================================================
# MODEL FACTORY
# ============================================================

class IndustrialModelFactory:
    @staticmethod
    def classification_models(
        seed: int,
    ) -> dict[str, Any]:
        return {
            "random_forest": (
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                )
            ),
            "hist_gradient_boosting": (
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    random_state=seed,
                )
            ),
        }

    @staticmethod
    def regression_models(
        seed: int,
    ) -> dict[str, Any]:
        return {
            "random_forest": (
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=14,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=seed,
                )
            ),
            "hist_gradient_boosting": (
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    random_state=seed,
                )
            ),
        }


# ============================================================
# EVALUATION
# ============================================================

class ModelEvaluator:
    @staticmethod
    def classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> dict[str, Any]:
        metrics = {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "balanced_accuracy": (
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "confusion_matrix": (
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                ).tolist()
            ),
        }

        if (
            probabilities is not None
            and len(np.unique(y_true)) == 2
        ):
            metrics["roc_auc"] = (
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            )

            metrics["average_precision"] = (
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

        return json_safe(metrics)

    @staticmethod
    def regression(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        mse = mean_squared_error(
            y_true,
            y_pred,
        )

        return json_safe({
            "mae": mean_absolute_error(
                y_true,
                y_pred,
            ),
            "rmse": math.sqrt(mse),
            "r2": (
                r2_score(
                    y_true,
                    y_pred,
                )
                if len(y_true) >= 2
                else None
            ),
        })


# ============================================================
# TRAINER
# ============================================================

class IndustrialModelTrainer:
    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        self.config = config

        self.output_dir = Path(
            config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.feature_engineer = (
            IndustrialFeatureEngineer()
        )

        self.splitter = DatasetSplitter(
            config
        )

        self.evaluator = ModelEvaluator()

        self.mlflow = None

        if config.enable_mlflow:
            try:
                import mlflow

                self.mlflow = mlflow

                mlflow.set_experiment(
                    config.mlflow_experiment
                )
            except ImportError as exc:
                raise RuntimeError(
                    "MLflow is not installed"
                ) from exc

        self._validate_config()

    def _validate_config(self) -> None:
        if not (
            0 < self.config.test_size < 1
        ):
            raise ValueError(
                "test_size must be between 0 and 1"
            )

        if not (
            0 < self.config.validation_size < 1
        ):
            raise ValueError(
                "validation_size must be between 0 and 1"
            )

        if (
            self.config.test_size
            + self.config.validation_size
            >= 1
        ):
            raise ValueError(
                "Combined validation and test "
                "sizes must be less than 1"
            )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    @staticmethod
    def make_pipeline(
        model: Any,
        feature_names: list[str],
    ) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                    feature_names,
                )
            ],
            remainder="drop",
        )

        return Pipeline([
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ])

    # --------------------------------------------------------
    # CLASSIFICATION TRAINING
    # --------------------------------------------------------

    def train_classifier(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        task_name: str,
    ) -> dict[str, Any]:
        if target_column not in dataframe.columns:
            raise ValueError(
                f"Missing target: {target_column}"
            )

        data = dataframe.dropna(
            subset=[target_column]
        ).reset_index(drop=True)

        data[target_column] = pd.to_numeric(
            data[target_column],
            errors="raise",
        ).astype(int)

        if not set(
            data[target_column].unique()
        ).issubset({0, 1}):
            raise ValueError(
                "Classification target must be binary"
            )

        if len(data) < self.config.min_training_rows:
            raise ValueError(
                "Insufficient training rows"
            )

        X = self.feature_engineer.transform(
            data
        )

        y = data[
            target_column
        ].to_numpy()

        train_idx, val_idx, test_idx = (
            self.splitter.split(
                data
            )
        )

        if len(np.unique(y[train_idx])) < 2:
            raise ValueError(
                "Training partition contains only "
                "one class. Increase events/assets "
                "or adjust the split."
            )

        models = (
            IndustrialModelFactory
            .classification_models(
                self.config.seed
            )
        )

        candidates = []

        for model_name, estimator in models.items():
            LOGGER.info(
                "Training %s / %s",
                task_name,
                model_name,
            )

            pipeline = self.make_pipeline(
                estimator,
                X.columns.tolist(),
            )

            pipeline.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            val_pred = pipeline.predict(
                X.iloc[val_idx]
            )

            val_prob = (
                pipeline.predict_proba(
                    X.iloc[val_idx]
                )[:, 1]
                if hasattr(
                    pipeline,
                    "predict_proba",
                )
                else None
            )

            metrics = (
                self.evaluator.classification(
                    y[val_idx],
                    val_pred,
                    val_prob,
                )
            )

            candidates.append({
                "name": model_name,
                "pipeline": pipeline,
                "validation_metrics": metrics,
            })

        best = max(
            candidates,
            key=lambda candidate: (
                candidate[
                    "validation_metrics"
                ]["average_precision"]
                if candidate[
                    "validation_metrics"
                ]["average_precision"]
                is not None
                else candidate[
                    "validation_metrics"
                ]["f1"]
            ),
        )

        best_pipeline = best[
            "pipeline"
        ]

        # The test set remains untouched during
        # model selection.
        test_pred = best_pipeline.predict(
            X.iloc[test_idx]
        )

        test_prob = (
            best_pipeline.predict_proba(
                X.iloc[test_idx]
            )[:, 1]
            if hasattr(
                best_pipeline,
                "predict_proba",
            )
            else None
        )

        test_metrics = (
            self.evaluator.classification(
                y[test_idx],
                test_pred,
                test_prob,
            )
        )

        task_dir = (
            self.output_dir / task_name
        )

        task_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            task_dir / "model.joblib"
        )

        joblib.dump(
            best_pipeline,
            model_path,
        )

        report = {
            "task": task_name,
            "target": target_column,
            "model_name": best["name"],
            "trained_at": utc_now(),
            "training_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "test_rows": len(test_idx),
            "positive_train": int(
                y[train_idx].sum()
            ),
            "positive_validation": int(
                y[val_idx].sum()
            ),
            "positive_test": int(
                y[test_idx].sum()
            ),
            "features": X.columns.tolist(),
            "split_strategy": (
                self.config.split_strategy
            ),
            "validation_metrics": (
                best["validation_metrics"]
            ),
            "test_metrics": test_metrics,
            "candidate_metrics": {
                candidate["name"]: candidate[
                    "validation_metrics"
                ]
                for candidate in candidates
            },
            "model_path": str(
                model_path
            ),
            "synthetic_data": True,
        }

        save_json(
            task_dir / "metrics.json",
            report,
        )

        self._log_mlflow(
            task_name,
            best["name"],
            test_metrics,
            model_path,
        )

        return report

    # --------------------------------------------------------
    # RUL REGRESSION
    # --------------------------------------------------------

    def train_regressor(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        task_name: str,
    ) -> dict[str, Any]:
        if target_column not in dataframe.columns:
            raise ValueError(
                f"Missing target: {target_column}"
            )

        data = dataframe.copy()

        data[target_column] = pd.to_numeric(
            data[target_column],
            errors="coerce",
        )

        if "rul_observed" in data.columns:
            observed = (
                data["rul_observed"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
            )

            data = data.loc[
                observed
            ].copy()

        data = data.dropna(
            subset=[target_column]
        ).reset_index(drop=True)

        data = data.loc[
            data[target_column] >= 0
        ].reset_index(drop=True)

        if len(data) < self.config.min_training_rows:
            raise ValueError(
                "Insufficient observed RUL labels"
            )

        X = self.feature_engineer.transform(
            data
        )

        y = data[
            target_column
        ].to_numpy(dtype=float)

        train_idx, val_idx, test_idx = (
            self.splitter.split(
                data
            )
        )

        models = (
            IndustrialModelFactory
            .regression_models(
                self.config.seed
            )
        )

        candidates = []

        for model_name, estimator in models.items():
            LOGGER.info(
                "Training %s / %s",
                task_name,
                model_name,
            )

            pipeline = self.make_pipeline(
                estimator,
                X.columns.tolist(),
            )

            pipeline.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            predictions = pipeline.predict(
                X.iloc[val_idx]
            )

            predictions = np.maximum(
                predictions,
                0,
            )

            metrics = (
                self.evaluator.regression(
                    y[val_idx],
                    predictions,
                )
            )

            candidates.append({
                "name": model_name,
                "pipeline": pipeline,
                "validation_metrics": metrics,
            })

        best = min(
            candidates,
            key=lambda candidate: (
                candidate[
                    "validation_metrics"
                ]["mae"]
            ),
        )

        test_predictions = best[
            "pipeline"
        ].predict(
            X.iloc[test_idx]
        )

        test_predictions = np.maximum(
            test_predictions,
            0,
        )

        test_metrics = (
            self.evaluator.regression(
                y[test_idx],
                test_predictions,
            )
        )

        task_dir = (
            self.output_dir / task_name
        )

        task_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            task_dir / "model.joblib"
        )

        joblib.dump(
            best["pipeline"],
            model_path,
        )

        report = {
            "task": task_name,
            "target": target_column,
            "model_name": best["name"],
            "trained_at": utc_now(),
            "training_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "test_rows": len(test_idx),
            "features": X.columns.tolist(),
            "split_strategy": (
                self.config.split_strategy
            ),
            "validation_metrics": (
                best["validation_metrics"]
            ),
            "test_metrics": test_metrics,
            "candidate_metrics": {
                candidate["name"]: candidate[
                    "validation_metrics"
                ]
                for candidate in candidates
            },
            "model_path": str(
                model_path
            ),
            "synthetic_data": True,
            "limitation": (
                "Only observed RUL labels are "
                "used; censoring introduces "
                "selection bias."
            ),
        }

        save_json(
            task_dir / "metrics.json",
            report,
        )

        self._log_mlflow(
            task_name,
            best["name"],
            test_metrics,
            model_path,
        )

        return report

    # --------------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------------

    def train_anomaly_model(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Train supervised anomaly classifiers.

        Also train an IsolationForest baseline
        on nominal training observations only.
        """

        report = self.train_classifier(
            dataframe=dataframe,
            target_column="anomaly_label",
            task_name="anomaly",
        )

        data = dataframe.dropna(
            subset=["anomaly_label"]
        ).reset_index(drop=True)

        X = self.feature_engineer.transform(
            data
        )

        y = pd.to_numeric(
            data["anomaly_label"],
            errors="raise",
        ).astype(int).to_numpy()

        train_idx, val_idx, test_idx = (
            self.splitter.split(
                data
            )
        )

        normal_indices = train_idx[
            y[train_idx] == 0
        ]

        if len(normal_indices) < 10:
            LOGGER.warning(
                "Insufficient nominal readings "
                "for IsolationForest"
            )

            return report

        pipeline = self.make_pipeline(
            IsolationForest(
                n_estimators=200,
                contamination=0.025,
                random_state=(
                    self.config.seed
                ),
                n_jobs=-1,
            ),
            X.columns.tolist(),
        )

        pipeline.fit(
            X.iloc[normal_indices]
        )

        raw_predictions = pipeline.predict(
            X.iloc[test_idx]
        )

        predictions = (
            raw_predictions == -1
        ).astype(int)

        metrics = (
            self.evaluator.classification(
                y[test_idx],
                predictions,
            )
        )

        task_dir = (
            self.output_dir / "anomaly"
        )

        isolation_path = (
            task_dir
            / "isolation_forest.joblib"
        )

        joblib.dump(
            pipeline,
            isolation_path,
        )

        report["isolation_forest"] = {
            "metrics": metrics,
            "model_path": str(
                isolation_path
            ),
        }

        save_json(
            task_dir / "metrics.json",
            report,
        )

        return report

    # --------------------------------------------------------
    # DATASET LOADING
    # --------------------------------------------------------

    def load_anomaly_data(
        self,
    ) -> pd.DataFrame:
        path = (
            Path(self.config.data_dir)
            / "telemetry"
            / "telemetry.csv"
        )

        return load_dataset(
            path,
            self.config.max_rows,
        )

    def load_failure_data(
        self,
    ) -> pd.DataFrame:
        path = (
            Path(self.config.data_dir)
            / "failures"
            / "failure_predictions.csv"
        )

        return load_dataset(
            path,
            self.config.max_rows,
        )

    def load_rul_data(
        self,
    ) -> pd.DataFrame:
        root = (
            Path(self.config.data_dir)
            / "failures"
        )

        prediction_path = (
            root / "failure_predictions.csv"
        )

        rul_path = (
            root / "rul_labels.csv"
        )

        predictions = load_dataset(
            prediction_path,
            self.config.max_rows,
        )

        rul = load_dataset(
            rul_path,
            self.config.max_rows,
        )

        merged = predictions.merge(
            rul[
                [
                    "reading_id",
                    "rul_hours",
                    "rul_observed",
                ]
            ],
            on="reading_id",
            how="inner",
            validate="one_to_one",
        )

        return merged

    # --------------------------------------------------------
    # MLFLOW
    # --------------------------------------------------------

    def _log_mlflow(
        self,
        task_name: str,
        model_name: str,
        metrics: dict[str, Any],
        model_path: Path,
    ) -> None:
        if self.mlflow is None:
            return

        with self.mlflow.start_run(
            run_name=(
                f"{task_name}-{model_name}"
            )
        ):
            self.mlflow.log_params({
                "task": task_name,
                "model": model_name,
                "seed": self.config.seed,
                "split_strategy": (
                    self.config.split_strategy
                ),
            })

            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(
                    value,
                    (int, float),
                )
                and value is not None
                and math.isfinite(
                    float(value)
                )
            }

            self.mlflow.log_metrics(
                numeric_metrics
            )

            self.mlflow.log_artifact(
                str(model_path)
            )

    # --------------------------------------------------------
    # TRAIN ALL
    # --------------------------------------------------------

    def run(
        self,
    ) -> dict[str, Any]:
        tasks = (
            ["anomaly", "failure", "rul"]
            if self.config.task == "all"
            else [self.config.task]
        )

        results = {}

        for task in tasks:
            try:
                LOGGER.info(
                    "Starting task: %s",
                    task,
                )

                if task == "anomaly":
                    dataframe = (
                        self.load_anomaly_data()
                    )

                    result = (
                        self.train_anomaly_model(
                            dataframe
                        )
                    )

                elif task == "failure":
                    dataframe = (
                        self.load_failure_data()
                    )

                    result = self.train_classifier(
                        dataframe,
                        target_column=(
                            "failure_within_horizon"
                        ),
                        task_name="failure",
                    )

                elif task == "rul":
                    dataframe = (
                        self.load_rul_data()
                    )

                    result = self.train_regressor(
                        dataframe,
                        target_column="rul_hours",
                        task_name="rul",
                    )

                else:
                    raise ValueError(
                        f"Unsupported task: {task}"
                    )

                results[task] = {
                    "status": "success",
                    "report": result,
                }

            except Exception as exc:
                LOGGER.exception(
                    "Training failed for %s",
                    task,
                )

                results[task] = {
                    "status": "failed",
                    "error": str(exc),
                }

        summary = {
            "generated_at": utc_now(),
            "configuration": asdict(
                self.config
            ),
            "tasks": results,
        }

        save_json(
            self.output_dir
            / "training_summary.json",
            summary,
        )

        return summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train industrial predictive "
            "maintenance ML models"
        )
    )

    parser.add_argument(
        "--task",
        choices=[
            "anomaly",
            "failure",
            "rul",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--data-dir",
        default="data/synthetic",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/models",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--split-strategy",
        choices=["asset", "time"],
        default="asset",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--tune",
        action="store_true",
    )

    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--mlflow",
        action="store_true",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task=args.task,
        seed=args.seed,
        split_strategy=(
            args.split_strategy
        ),
        test_size=args.test_size,
        validation_size=(
            args.validation_size
        ),
        tune=args.tune,
        tuning_iterations=(
            args.tuning_iterations
        ),
        enable_mlflow=args.mlflow,
        max_rows=args.max_rows,
    )

    trainer = IndustrialModelTrainer(
        config
    )

    summary = trainer.run()

    print(
        json.dumps(
            json_safe(summary),
            indent=2,
        )
    )

    if any(
        result["status"] == "failed"
        for result in summary["tasks"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()