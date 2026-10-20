"""
app/ai/ml/models/financial_forecast.py

Industrial Financial Forecasting Intelligence
=============================================

Python 3.11+

Dependencies:
    pip install numpy pandas scikit-learn pydantic

Features:
- Historical industrial cost analysis
- Monthly financial forecasting
- Maintenance cost prediction
- Repair and replacement cost forecasting
- Downtime and production loss forecasting
- Spare parts, labor, energy and emergency costs
- Ridge regression forecasting
- Seasonal naive forecasting
- Time-based backtesting
- MAE, RMSE and sMAPE evaluation
- Prediction intervals from historical residuals
- Optimistic, baseline and pessimistic scenarios
- Budget variance analysis
- Multi-tenant asset forecasting
- Model persistence
- JSON-compatible results

Important:
- Predictions are estimates, not guarantees.
- Prediction intervals are empirical, not
  formally calibrated guarantees.
- Historical records should represent
  non-overlapping cost categories.
- Only load trusted joblib artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# EXCEPTIONS
# ============================================================

class FinancialForecastError(Exception):
    pass


class InsufficientFinancialDataError(
    FinancialForecastError
):
    pass


class FinancialModelNotFittedError(
    FinancialForecastError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class ForecastAlgorithm(str, Enum):
    RIDGE = "ridge"
    SEASONAL_NAIVE = "seasonal_naive"


class FinancialCostCategory(str, Enum):
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    SPARE_PARTS = "spare_parts"
    LABOR = "labor"
    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"
    ENERGY = "energy"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    OTHER = "other"


class ForecastScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    BASELINE = "baseline"
    PESSIMISTIC = "pessimistic"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class FinancialForecastConfig:
    algorithm: ForecastAlgorithm = (
        ForecastAlgorithm.RIDGE
    )

    date_column: str = "date"
    amount_column: str = "amount"
    category_column: str = "cost_category"

    currency: str = "INR"

    forecast_months: int = 12

    minimum_history_months: int = 12

    lag_months: int = 3

    ridge_alpha: float = 10.0

    interval_coverage: float = 0.90

    optimistic_multiplier: float = 0.85

    pessimistic_multiplier: float = 1.20

    random_state: int = 42

    def __post_init__(self) -> None:
        if self.forecast_months < 1:
            raise ValueError(
                "forecast_months must be positive"
            )

        if self.minimum_history_months < 3:
            raise ValueError(
                "minimum_history_months must be >= 3"
            )

        if self.lag_months < 1:
            raise ValueError(
                "lag_months must be >= 1"
            )

        if not 0 < self.interval_coverage < 1:
            raise ValueError(
                "interval_coverage must be in (0, 1)"
            )

        if self.ridge_alpha < 0:
            raise ValueError(
                "ridge_alpha must be nonnegative"
            )

        if not (
            0 < self.optimistic_multiplier <= 1
        ):
            raise ValueError(
                "optimistic_multiplier must be in (0, 1]"
            )

        if self.pessimistic_multiplier < 1:
            raise ValueError(
                "pessimistic_multiplier must be >= 1"
            )

        if (
            len(self.currency) != 3
            or not self.currency.isalpha()
            or self.currency != self.currency.upper()
        ):
            raise ValueError(
                "currency must be an uppercase "
                "three-letter code"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class MonthlyFinancialForecast:
    month: str

    predicted_cost: float

    lower_bound: float
    upper_bound: float

    optimistic_cost: float
    pessimistic_cost: float

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "predicted_cost": round(
                self.predicted_cost, 2
            ),
            "lower_bound": round(
                self.lower_bound, 2
            ),
            "upper_bound": round(
                self.upper_bound, 2
            ),
            "optimistic_cost": round(
                self.optimistic_cost, 2
            ),
            "pessimistic_cost": round(
                self.pessimistic_cost, 2
            ),
            "currency": self.currency,
        }


@dataclass
class FinancialForecastResult:
    forecast_id: str

    company_id: str
    asset_id: str

    category: str

    algorithm: str

    currency: str

    forecast_horizon_months: int

    monthly_forecasts: list[
        MonthlyFinancialForecast
    ]

    total_forecast_cost: float

    total_optimistic_cost: float
    total_pessimistic_cost: float

    generated_at: str

    model_version: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "company_id": self.company_id,
            "asset_id": self.asset_id,
            "category": self.category,
            "algorithm": self.algorithm,
            "currency": self.currency,
            "forecast_horizon_months": (
                self.forecast_horizon_months
            ),
            "monthly_forecasts": [
                item.to_dict()
                for item in self.monthly_forecasts
            ],
            "total_forecast_cost": round(
                self.total_forecast_cost, 2
            ),
            "total_optimistic_cost": round(
                self.total_optimistic_cost, 2
            ),
            "total_pessimistic_cost": round(
                self.total_pessimistic_cost, 2
            ),
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }


@dataclass
class FinancialEvaluationResult:
    sample_count: int

    mae: float
    rmse: float
    smape_percent: float

    actual_total: float
    predicted_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "smape_percent": round(
                self.smape_percent, 4
            ),
            "actual_total": round(
                self.actual_total, 2
            ),
            "predicted_total": round(
                self.predicted_total, 2
            ),
        }


@dataclass
class BudgetVarianceResult:
    budget_amount: float
    forecast_amount: float

    variance_amount: float
    variance_percent: Optional[float]

    over_budget: bool

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_amount": round(
                self.budget_amount, 2
            ),
            "forecast_amount": round(
                self.forecast_amount, 2
            ),
            "variance_amount": round(
                self.variance_amount, 2
            ),
            "variance_percent": (
                round(self.variance_percent, 2)
                if self.variance_percent is not None
                else None
            ),
            "over_budget": self.over_budget,
            "currency": self.currency,
        }


# ============================================================
# FINANCIAL DATA PROCESSOR
# ============================================================

class FinancialDataProcessor:

    def __init__(
        self,
        config: FinancialForecastConfig,
    ):
        self.config = config

    def prepare(
        self,
        data: pd.DataFrame,
        *,
        category: str | None = None,
    ) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame"
            )

        required = {
            self.config.date_column,
            self.config.amount_column,
        }

        missing = required - set(data.columns)

        if missing:
            raise FinancialForecastError(
                f"Missing columns: {sorted(missing)}"
            )

        df = data.copy()

        if category is not None:
            if (
                self.config.category_column
                not in df.columns
            ):
                raise FinancialForecastError(
                    "Category column is required "
                    "for category filtering"
                )

            df = df.loc[
                df[self.config.category_column]
                == category
            ].copy()

        if df.empty:
            raise InsufficientFinancialDataError(
                "No financial records available"
            )

        df[self.config.date_column] = (
            pd.to_datetime(
                df[self.config.date_column],
                errors="raise",
                utc=True,
            )
        )

        df[self.config.amount_column] = (
            pd.to_numeric(
                df[self.config.amount_column],
                errors="raise",
            )
        )

        amounts = df[
            self.config.amount_column
        ].to_numpy(dtype=float)

        if not np.isfinite(amounts).all():
            raise FinancialForecastError(
                "Cost amounts must be finite"
            )

        if (amounts < 0).any():
            raise FinancialForecastError(
                "Negative costs are unsupported; "
                "record credits separately"
            )

        df["month"] = (
            df[self.config.date_column]
            .dt.tz_convert(None)
            .dt.to_period("M")
        )

        monthly = (
            df.groupby("month")[
                self.config.amount_column
            ]
            .sum()
            .sort_index()
        )

        # Fill genuine gaps with zero cost.
        # The caller must ensure the underlying
        # transaction history is complete.
        full_index = pd.period_range(
            start=monthly.index.min(),
            end=monthly.index.max(),
            freq="M",
        )

        monthly = monthly.reindex(
            full_index,
            fill_value=0.0,
        )

        if (
            len(monthly)
            < self.config.minimum_history_months
        ):
            raise InsufficientFinancialDataError(
                "Insufficient monthly history: "
                f"{len(monthly)} available, "
                f"{self.config.minimum_history_months} "
                "required"
            )

        return monthly.astype(float)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class FinancialFeatureEngineer:

    def __init__(
        self,
        lag_months: int,
    ):
        self.lag_months = lag_months

    def make_features(
        self,
        values: np.ndarray,
        month_number: int,
    ) -> np.ndarray:
        if len(values) < self.lag_months:
            raise InsufficientFinancialDataError(
                "Not enough history for lag features"
            )

        recent = values[-self.lag_months:]

        month_of_year = (
            month_number % 12
        ) + 1

        features = [
            float(month_number),
            float(
                np.sin(
                    2 * np.pi
                    * month_of_year / 12
                )
            ),
            float(
                np.cos(
                    2 * np.pi
                    * month_of_year / 12
                )
            ),
            float(np.mean(recent)),
            float(np.std(recent)),
        ]

        features.extend(
            float(value)
            for value in recent
        )

        return np.asarray(
            features,
            dtype=float,
        )

    def supervised_dataset(
        self,
        monthly: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = monthly.to_numpy(
            dtype=float
        )

        X = []
        y = []

        for index in range(
            self.lag_months,
            len(values),
        ):
            X.append(
                self.make_features(
                    values[:index],
                    index,
                )
            )

            y.append(values[index])

        return (
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
        )


# ============================================================
# MAIN FINANCIAL FORECAST MODEL
# ============================================================

class IndustrialFinancialForecastModel:

    MODEL_VERSION = "1.0.0"

    def __init__(
        self,
        config: FinancialForecastConfig | None = None,
    ):
        self.config = (
            config or FinancialForecastConfig()
        )

        self.processor = FinancialDataProcessor(
            self.config
        )

        self.feature_engineer = (
            FinancialFeatureEngineer(
                self.config.lag_months
            )
        )

        self.model: Pipeline | None = None

        self.monthly_history: pd.Series | None = None

        self.residuals: np.ndarray | None = None

        self.is_fitted = False

        self.trained_at: str | None = None

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        *,
        category: str | None = None,
    ) -> "IndustrialFinancialForecastModel":

        monthly = self.processor.prepare(
            data,
            category=category,
        )

        self.monthly_history = monthly.copy()

        if (
            self.config.algorithm
            == ForecastAlgorithm.RIDGE
        ):
            X, y = (
                self.feature_engineer
                .supervised_dataset(monthly)
            )

            if len(y) < 5:
                raise InsufficientFinancialDataError(
                    "At least five supervised "
                    "training samples are required"
                )

            self.model = Pipeline([
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "regressor",
                    Ridge(
                        alpha=(
                            self.config.ridge_alpha
                        ),
                    ),
                ),
            ])

            self.model.fit(X, y)

            fitted = self.model.predict(X)

            self.residuals = (
                y - np.maximum(fitted, 0)
            )

        elif (
            self.config.algorithm
            == ForecastAlgorithm.SEASONAL_NAIVE
        ):
            self.model = None

            values = monthly.to_numpy(
                dtype=float
            )

            if len(values) > 12:
                self.residuals = (
                    values[12:]
                    - values[:-12]
                )
            else:
                self.residuals = np.diff(
                    values
                )

        else:
            raise FinancialForecastError(
                "Unsupported forecasting algorithm"
            )

        self.is_fitted = True

        self.trained_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        return self

    # --------------------------------------------------------
    # PREDICT NEXT MONTH
    # --------------------------------------------------------

    def _predict_next(
        self,
        history: np.ndarray,
        month_number: int,
    ) -> float:
        if (
            self.config.algorithm
            == ForecastAlgorithm.SEASONAL_NAIVE
        ):
            if len(history) >= 12:
                value = history[-12]
            else:
                value = history[-1]

        else:
            if self.model is None:
                raise FinancialModelNotFittedError(
                    "Regression model unavailable"
                )

            features = (
                self.feature_engineer
                .make_features(
                    history,
                    month_number,
                )
            )

            value = self.model.predict(
                features.reshape(1, -1)
            )[0]

        return max(
            0.0,
            float(value),
        )

    # --------------------------------------------------------
    # PREDICTION INTERVAL
    # --------------------------------------------------------

    def _interval_width(self) -> float:
        if (
            self.residuals is None
            or len(self.residuals) == 0
        ):
            return 0.0

        absolute_errors = np.abs(
            self.residuals
        )

        return float(
            np.quantile(
                absolute_errors,
                self.config.interval_coverage,
            )
        )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    def forecast(
        self,
        *,
        company_id: str,
        asset_id: str,
        category: str = "total",
        months: int | None = None,
    ) -> FinancialForecastResult:

        if (
            not self.is_fitted
            or self.monthly_history is None
        ):
            raise FinancialModelNotFittedError(
                "Fit the model before forecasting"
            )

        if not company_id or not asset_id:
            raise ValueError(
                "company_id and asset_id are required"
            )

        horizon = (
            months
            if months is not None
            else self.config.forecast_months
        )

        if horizon < 1:
            raise ValueError(
                "months must be positive"
            )

        history = list(
            self.monthly_history.to_numpy(
                dtype=float
            )
        )

        last_month = (
            self.monthly_history.index[-1]
        )

        base_length = len(history)

        interval_width = (
            self._interval_width()
        )

        monthly_results = []

        for step in range(
            1,
            horizon + 1,
        ):
            predicted = self._predict_next(
                np.asarray(
                    history,
                    dtype=float,
                ),
                base_length + step - 1,
            )

            # Residual uncertainty grows with
            # forecast horizon. This is a
            # heuristic, not a calibrated interval.
            width = (
                interval_width
                * np.sqrt(step)
            )

            lower = max(
                0.0,
                predicted - width,
            )

            upper = (
                predicted + width
            )

            optimistic = (
                predicted
                * self.config.optimistic_multiplier
            )

            pessimistic = (
                predicted
                * self.config.pessimistic_multiplier
            )

            forecast_month = (
                last_month + step
            )

            monthly_results.append(
                MonthlyFinancialForecast(
                    month=str(forecast_month),
                    predicted_cost=predicted,
                    lower_bound=lower,
                    upper_bound=upper,
                    optimistic_cost=optimistic,
                    pessimistic_cost=pessimistic,
                    currency=self.config.currency,
                )
            )

            history.append(predicted)

        return FinancialForecastResult(
            forecast_id=str(uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            category=category,
            algorithm=self.config.algorithm.value,
            currency=self.config.currency,
            forecast_horizon_months=horizon,
            monthly_forecasts=monthly_results,
            total_forecast_cost=sum(
                item.predicted_cost
                for item in monthly_results
            ),
            total_optimistic_cost=sum(
                item.optimistic_cost
                for item in monthly_results
            ),
            total_pessimistic_cost=sum(
                item.pessimistic_cost
                for item in monthly_results
            ),
            generated_at=datetime.now(
                timezone.utc
            ).isoformat(),
            model_version=self.MODEL_VERSION,
            metadata={
                "training_months": base_length,
                "trained_at": self.trained_at,
                "interval_method": (
                    "in_sample_absolute_residual_quantile"
                ),
                "interval_calibrated": False,
            },
        )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    def backtest(
        self,
        data: pd.DataFrame,
        *,
        category: str | None = None,
        test_months: int = 3,
    ) -> FinancialEvaluationResult:
        monthly = self.processor.prepare(
            data,
            category=category,
        )

        if test_months < 1:
            raise ValueError(
                "test_months must be positive"
            )

        train_count = (
            len(monthly) - test_months
        )

        required_train = max(
            self.config.minimum_history_months,
            self.config.lag_months + 5,
        )

        if train_count < required_train:
            raise InsufficientFinancialDataError(
                "Not enough history for "
                "the requested backtest"
            )

        actual = []
        predicted = []

        # Expanding-window one-step backtest.
        # Each evaluation month is predicted
        # using only preceding months.
        for index in range(
            train_count,
            len(monthly),
        ):
            train = monthly.iloc[:index]

            if (
                self.config.algorithm
                == ForecastAlgorithm.SEASONAL_NAIVE
            ):
                values = train.to_numpy(
                    dtype=float
                )

                estimate = (
                    values[-12]
                    if len(values) >= 12
                    else values[-1]
                )

            else:
                X, y = (
                    self.feature_engineer
                    .supervised_dataset(train)
                )

                temporary_model = Pipeline([
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "regressor",
                        Ridge(
                            alpha=self.config.ridge_alpha
                        ),
                    ),
                ])

                temporary_model.fit(X, y)

                features = (
                    self.feature_engineer
                    .make_features(
                        train.to_numpy(
                            dtype=float
                        ),
                        len(train),
                    )
                )

                estimate = (
                    temporary_model.predict(
                        features.reshape(1, -1)
                    )[0]
                )

            actual.append(
                float(monthly.iloc[index])
            )

            predicted.append(
                max(0.0, float(estimate))
            )

        actual_array = np.asarray(
            actual,
            dtype=float,
        )

        predicted_array = np.asarray(
            predicted,
            dtype=float,
        )

        mae = float(
            mean_absolute_error(
                actual_array,
                predicted_array,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    actual_array,
                    predicted_array,
                )
            )
        )

        denominator = (
            np.abs(actual_array)
            + np.abs(predicted_array)
        )

        smape_terms = np.divide(
            200.0
            * np.abs(
                actual_array - predicted_array
            ),
            denominator,
            out=np.zeros_like(
                denominator
            ),
            where=denominator > 1e-8,
        )

        return FinancialEvaluationResult(
            sample_count=len(actual),
            mae=mae,
            rmse=rmse,
            smape_percent=float(
                np.mean(smape_terms)
            ),
            actual_total=float(
                actual_array.sum()
            ),
            predicted_total=float(
                predicted_array.sum()
            ),
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        if not self.is_fitted:
            raise FinancialModelNotFittedError(
                "Cannot save an unfitted model"
            )

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "config": self.config,
                "model": self.model,
                "monthly_history": (
                    self.monthly_history
                ),
                "residuals": self.residuals,
                "trained_at": self.trained_at,
                "model_version": (
                    self.MODEL_VERSION
                ),
            },
            destination,
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "IndustrialFinancialForecastModel":
        # Joblib can execute code during loading.
        # Only load artifacts from trusted sources.
        payload = joblib.load(path)

        instance = cls(
            config=payload["config"]
        )

        instance.model = payload["model"]

        instance.monthly_history = (
            payload["monthly_history"]
        )

        instance.residuals = (
            payload["residuals"]
        )

        instance.trained_at = (
            payload["trained_at"]
        )

        instance.is_fitted = True

        return instance


# ============================================================
# BUDGET ANALYZER
# ============================================================

class IndustrialBudgetAnalyzer:

    @staticmethod
    def compare(
        *,
        budget_amount: float,
        forecast_amount: float,
        currency: str = "INR",
    ) -> BudgetVarianceResult:

        if budget_amount < 0:
            raise ValueError(
                "budget_amount cannot be negative"
            )

        if forecast_amount < 0:
            raise ValueError(
                "forecast_amount cannot be negative"
            )

        variance = (
            forecast_amount - budget_amount
        )

        variance_percent = None

        if budget_amount > 0:
            variance_percent = (
                variance / budget_amount
            ) * 100

        return BudgetVarianceResult(
            budget_amount=budget_amount,
            forecast_amount=forecast_amount,
            variance_amount=variance,
            variance_percent=variance_percent,
            over_budget=variance > 0,
            currency=currency,
        )


# ============================================================
# MULTI-CATEGORY FINANCIAL FORECAST
# ============================================================

class IndustrialFinancialForecastService:
    """
    Coordinates independent cost forecasts.

    One model is trained per cost category.

    Do not combine forecasts across different
    companies, assets or currencies without
    explicit filtering and conversion.
    """

    def __init__(
        self,
        config: FinancialForecastConfig | None = None,
    ):
        self.config = (
            config or FinancialForecastConfig()
        )

        self.models: dict[
            str,
            IndustrialFinancialForecastModel
        ] = {}

    def fit_categories(
        self,
        data: pd.DataFrame,
        *,
        categories: list[str],
    ) -> dict[str, str]:

        results = {}

        for category in categories:
            try:
                model = (
                    IndustrialFinancialForecastModel(
                        self.config
                    )
                )

                model.fit(
                    data,
                    category=category,
                )

                self.models[category] = model

                results[category] = "trained"

            except FinancialForecastError as exc:
                results[category] = (
                    f"failed: {exc}"
                )

        return results

    def forecast_categories(
        self,
        *,
        company_id: str,
        asset_id: str,
        months: int | None = None,
    ) -> dict[str, FinancialForecastResult]:

        return {
            category: model.forecast(
                company_id=company_id,
                asset_id=asset_id,
                category=category,
                months=months,
            )
            for category, model in self.models.items()
        }


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    rng = np.random.default_rng(42)

    months = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="MS",
    )

    # Synthetic financial history for testing.
    history = pd.DataFrame({
        "date": months,
        "amount": np.maximum(
            0,
            50000
            + np.arange(30) * 800
            + 7000 * np.sin(
                np.arange(30)
                * 2 * np.pi / 12
            )
            + rng.normal(
                0,
                3000,
                30,
            ),
        ),
        "cost_category": [
            "maintenance"
        ] * 30,
    })

    config = FinancialForecastConfig(
        algorithm=ForecastAlgorithm.RIDGE,
        forecast_months=6,
        minimum_history_months=12,
        lag_months=3,
        currency="INR",
    )

    model = IndustrialFinancialForecastModel(
        config
    )

    evaluation = model.backtest(
        history,
        category="maintenance",
        test_months=6,
    )

    print("BACKTEST")

    print(
        json.dumps(
            evaluation.to_dict(),
            indent=2,
        )
    )

    model.fit(
        history,
        category="maintenance",
    )

    forecast = model.forecast(
        company_id="company-001",
        asset_id="ASSET-101",
        category="maintenance",
        months=6,
    )

    print("FINANCIAL FORECAST")

    print(
        json.dumps(
            forecast.to_dict(),
            indent=2,
        )
    )

    budget = IndustrialBudgetAnalyzer.compare(
        budget_amount=400000,
        forecast_amount=(
            forecast.total_forecast_cost
        ),
        currency="INR",
    )

    print("BUDGET ANALYSIS")

    print(
        json.dumps(
            budget.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()