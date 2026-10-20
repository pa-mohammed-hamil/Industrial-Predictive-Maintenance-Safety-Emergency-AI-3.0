"""
Industrial Time-Series Forecasting
==================================

Unified forecasting engine for industrial telemetry.

Supports:
- univariate and multivariate forecasting
- trend-aware forecasts
- autoregressive lag forecasts
- moving-average forecasts
- exponential smoothing
- ensemble forecasting
- multi-step horizons
- prediction intervals
- degradation-aware forecasting
- forecast risk scoring
- future timestamps
- model evaluation
- model selection

Designed to integrate with:
    feature_engineering.py
    forecasting_utils.py
    sequence_models.py
    transformer_models.py
    anomaly_detection/
    degradation_model.py
    failure_prediction.py
    rul_prediction.py
    maintenance_forecasting.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# =====================================================================
# Enums
# =====================================================================


class ForecastMethod(str, Enum):
    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_TREND = "linear_trend"
    AUTOREGRESSIVE = "autoregressive"
    ENSEMBLE = "ensemble"


class ForecastTrend(str, Enum):
    RAPIDLY_INCREASING = "rapidly_increasing"
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    RAPIDLY_DECREASING = "rapidly_decreasing"


class ForecastRiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class ForecastThreshold:
    healthy_min: Optional[float] = None
    healthy_max: Optional[float] = None
    warning_min: Optional[float] = None
    warning_max: Optional[float] = None
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None


@dataclass
class ForecastConfig:
    method: ForecastMethod = ForecastMethod.ENSEMBLE

    horizon: int = 24

    moving_average_window: int = 12
    trend_window: int = 24

    exponential_alpha: float = 0.30

    autoregressive_lags: Sequence[int] = (
        1,
        2,
        3,
        6,
        12,
        24,
    )

    confidence_level: float = 0.95

    min_history: int = 12

    ensemble_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "naive": 0.10,
            "moving_average": 0.15,
            "exponential_smoothing": 0.20,
            "linear_trend": 0.25,
            "autoregressive": 0.30,
        }
    )

    thresholds: Mapping[str, ForecastThreshold] = field(
        default_factory=dict
    )


# =====================================================================
# Result objects
# =====================================================================


@dataclass
class ForecastPoint:
    step: int
    timestamp: Optional[datetime]
    predicted_value: float
    lower_bound: float
    upper_bound: float
    risk_score: float
    risk_level: ForecastRiskLevel

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": (
                self.timestamp.isoformat()
                if self.timestamp
                else None
            ),
            "predicted_value": round(
                self.predicted_value, 6
            ),
            "lower_bound": round(
                self.lower_bound, 6
            ),
            "upper_bound": round(
                self.upper_bound, 6
            ),
            "risk_score": round(
                self.risk_score, 6
            ),
            "risk_level": self.risk_level.value,
        }


@dataclass
class ForecastResult:
    signal_name: str
    generated_at: datetime

    method: ForecastMethod
    horizon: int

    last_observation: float
    trend: ForecastTrend
    trend_slope: float

    confidence: float

    forecast_mean: float
    forecast_min: float
    forecast_max: float

    peak_risk_score: float
    peak_risk_level: ForecastRiskLevel

    threshold_crossing_step: Optional[int]

    points: list[ForecastPoint]

    model_details: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "generated_at": self.generated_at.isoformat(),
            "method": self.method.value,
            "horizon": self.horizon,
            "last_observation": round(
                self.last_observation, 6
            ),
            "trend": self.trend.value,
            "trend_slope": round(
                self.trend_slope, 6
            ),
            "confidence": round(
                self.confidence, 6
            ),
            "forecast_mean": round(
                self.forecast_mean, 6
            ),
            "forecast_min": round(
                self.forecast_min, 6
            ),
            "forecast_max": round(
                self.forecast_max, 6
            ),
            "peak_risk_score": round(
                self.peak_risk_score, 6
            ),
            "peak_risk_level": (
                self.peak_risk_level.value
            ),
            "threshold_crossing_step": (
                self.threshold_crossing_step
            ),
            "points": [
                point.to_dict()
                for point in self.points
            ],
            "model_details": self.model_details,
        }


# =====================================================================
# Forecaster
# =====================================================================


class IndustrialForecaster:
    """
    Deterministic industrial forecasting baseline.

    Intended as:
    - production fallback model
    - baseline for ML/DL evaluation
    - interpretable forecasting engine
    - short-horizon telemetry forecaster

    Trained sequence/transformer models can later sit behind the same
    service interface.
    """

    def __init__(
        self,
        config: Optional[ForecastConfig] = None,
    ) -> None:
        self.config = config or ForecastConfig()

        if self.config.horizon <= 0:
            raise ValueError(
                "Forecast horizon must be positive."
            )

        if not 0.0 < self.config.exponential_alpha <= 1.0:
            raise ValueError(
                "exponential_alpha must be in (0, 1]."
            )

        if not 0.0 < self.config.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be in (0, 1)."
            )

    # =================================================================
    # Public API
    # =================================================================

    def forecast(
        self,
        values: Sequence[float] | np.ndarray | pd.Series,
        *,
        signal_name: str = "signal",
        horizon: Optional[int] = None,
        timestamps: Optional[Sequence[Any]] = None,
        sampling_interval: Optional[
            str | pd.Timedelta
        ] = None,
        method: Optional[ForecastMethod | str] = None,
    ) -> ForecastResult:

        series = self._clean_values(values)

        if len(series) < 2:
            raise ValueError(
                "At least two finite observations are required."
            )

        forecast_horizon = int(
            horizon or self.config.horizon
        )

        if forecast_horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        selected_method = self._resolve_method(
            method or self.config.method
        )

        predictions, model_details = (
            self._forecast_by_method(
                series,
                forecast_horizon,
                selected_method,
            )
        )

        residual_std = self._estimate_residual_std(
            series,
            selected_method,
        )

        lower, upper = self._prediction_intervals(
            predictions,
            residual_std,
        )

        slope = self._trend_slope(
            series
        )

        trend = self._classify_trend(
            slope,
            series,
        )

        future_timestamps = self._future_timestamps(
            timestamps=timestamps,
            horizon=forecast_horizon,
            sampling_interval=sampling_interval,
        )

        threshold = self.config.thresholds.get(
            signal_name
        )

        points: list[ForecastPoint] = []

        threshold_crossing_step: Optional[int] = None

        for index, prediction in enumerate(
            predictions
        ):
            risk_score = self._forecast_risk(
                float(prediction),
                threshold,
            )

            risk_level = self._risk_level(
                risk_score
            )

            if (
                threshold_crossing_step is None
                and risk_score >= 0.45
            ):
                threshold_crossing_step = index + 1

            points.append(
                ForecastPoint(
                    step=index + 1,
                    timestamp=(
                        future_timestamps[index]
                        if future_timestamps
                        else None
                    ),
                    predicted_value=float(
                        prediction
                    ),
                    lower_bound=float(
                        lower[index]
                    ),
                    upper_bound=float(
                        upper[index]
                    ),
                    risk_score=risk_score,
                    risk_level=risk_level,
                )
            )

        peak_risk_score = max(
            (
                point.risk_score
                for point in points
            ),
            default=0.0,
        )

        confidence = self._confidence(
            series=series,
            residual_std=residual_std,
            predictions=predictions,
        )

        return ForecastResult(
            signal_name=signal_name,
            generated_at=datetime.now(
                timezone.utc
            ),
            method=selected_method,
            horizon=forecast_horizon,
            last_observation=float(
                series[-1]
            ),
            trend=trend,
            trend_slope=slope,
            confidence=confidence,
            forecast_mean=float(
                np.mean(predictions)
            ),
            forecast_min=float(
                np.min(predictions)
            ),
            forecast_max=float(
                np.max(predictions)
            ),
            peak_risk_score=peak_risk_score,
            peak_risk_level=self._risk_level(
                peak_risk_score
            ),
            threshold_crossing_step=(
                threshold_crossing_step
            ),
            points=points,
            model_details=model_details,
        )

    def forecast_dict(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:

        result = self.forecast(
            payload["values"],
            signal_name=payload.get(
                "signal_name",
                "signal",
            ),
            horizon=payload.get(
                "horizon"
            ),
            timestamps=payload.get(
                "timestamps"
            ),
            sampling_interval=payload.get(
                "sampling_interval"
            ),
            method=payload.get(
                "method"
            ),
        )

        return result.to_dict()

    def forecast_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        signal_columns: Sequence[str],
        timestamp_column: str = "timestamp",
        horizon: Optional[int] = None,
    ) -> dict[str, ForecastResult]:

        if dataframe.empty:
            raise ValueError(
                "dataframe cannot be empty."
            )

        timestamps = (
            dataframe[timestamp_column]
            if timestamp_column in dataframe.columns
            else None
        )

        results: dict[
            str,
            ForecastResult,
        ] = {}

        for signal in signal_columns:
            if signal not in dataframe.columns:
                raise ValueError(
                    f"Missing signal column: {signal}"
                )

            results[signal] = self.forecast(
                dataframe[signal],
                signal_name=signal,
                horizon=horizon,
                timestamps=timestamps,
            )

        return results

    # =================================================================
    # Forecast routing
    # =================================================================

    def _forecast_by_method(
        self,
        series: np.ndarray,
        horizon: int,
        method: ForecastMethod,
    ) -> tuple[np.ndarray, dict[str, Any]]:

        if method == ForecastMethod.NAIVE:
            return (
                self._naive(series, horizon),
                {"model": "last_value"},
            )

        if method == ForecastMethod.MOVING_AVERAGE:
            return (
                self._moving_average(
                    series,
                    horizon,
                ),
                {
                    "window":
                        self.config.moving_average_window
                },
            )

        if method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            return (
                self._exponential_smoothing(
                    series,
                    horizon,
                ),
                {
                    "alpha":
                        self.config.exponential_alpha
                },
            )

        if method == ForecastMethod.LINEAR_TREND:
            return (
                self._linear_trend_forecast(
                    series,
                    horizon,
                ),
                {
                    "trend_window":
                        self.config.trend_window
                },
            )

        if method == ForecastMethod.AUTOREGRESSIVE:
            prediction = (
                self._autoregressive_forecast(
                    series,
                    horizon,
                )
            )

            return (
                prediction,
                {
                    "lags": list(
                        self.config.autoregressive_lags
                    )
                },
            )

        return self._ensemble_forecast(
            series,
            horizon,
        )

    # =================================================================
    # Forecast models
    # =================================================================

    @staticmethod
    def _naive(
        series: np.ndarray,
        horizon: int,
    ) -> np.ndarray:

        return np.full(
            horizon,
            series[-1],
            dtype=float,
        )

    def _moving_average(
        self,
        series: np.ndarray,
        horizon: int,
    ) -> np.ndarray:

        window = min(
            self.config.moving_average_window,
            len(series),
        )

        history = list(
            series.astype(float)
        )

        predictions = []

        for _ in range(horizon):
            prediction = float(
                np.mean(
                    history[-window:]
                )
            )

            predictions.append(
                prediction
            )

            history.append(
                prediction
            )

        return np.asarray(
            predictions,
            dtype=float,
        )

    def _exponential_smoothing(
        self,
        series: np.ndarray,
        horizon: int,
    ) -> np.ndarray:

        alpha = (
            self.config.exponential_alpha
        )

        level = float(
            series[0]
        )

        for value in series[1:]:
            level = (
                alpha * float(value)
                + (1.0 - alpha) * level
            )

        return np.full(
            horizon,
            level,
            dtype=float,
        )

    def _linear_trend_forecast(
        self,
        series: np.ndarray,
        horizon: int,
    ) -> np.ndarray:

        window = min(
            self.config.trend_window,
            len(series),
        )

        values = series[
            -window:
        ]

        x = np.arange(
            len(values),
            dtype=float,
        )

        if len(values) < 2:
            return self._naive(
                series,
                horizon,
            )

        slope, intercept = np.polyfit(
            x,
            values,
            1,
        )

        future_x = np.arange(
            len(values),
            len(values) + horizon,
            dtype=float,
        )

        return (
            intercept
            + slope * future_x
        ).astype(float)

    def _autoregressive_forecast(
        self,
        series: np.ndarray,
        horizon: int,
    ) -> np.ndarray:

        configured_lags = sorted(
            {
                int(lag)
                for lag in self.config.autoregressive_lags
                if int(lag) > 0
            }
        )

        if not configured_lags:
            return self._naive(
                series,
                horizon,
            )

        max_lag = max(
            configured_lags
        )

        if len(series) <= max_lag + 2:
            return self._linear_trend_forecast(
                series,
                horizon,
            )

        X = []
        y = []

        for index in range(
            max_lag,
            len(series),
        ):
            row = [
                series[index - lag]
                for lag in configured_lags
            ]

            X.append(row)
            y.append(series[index])

        X_array = np.asarray(
            X,
            dtype=float,
        )

        y_array = np.asarray(
            y,
            dtype=float,
        )

        # Add intercept.
        design = np.column_stack(
            [
                np.ones(
                    len(X_array)
                ),
                X_array,
            ]
        )

        try:
            coefficients, *_ = np.linalg.lstsq(
                design,
                y_array,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return self._linear_trend_forecast(
                series,
                horizon,
            )

        history = list(
            series.astype(float)
        )

        predictions = []

        for _ in range(horizon):
            lag_values = [
                history[-lag]
                for lag in configured_lags
            ]

            row = np.asarray(
                [1.0, *lag_values],
                dtype=float,
            )

            prediction = float(
                row @ coefficients
            )

            if not np.isfinite(
                prediction
            ):
                prediction = float(
                    history[-1]
                )

            predictions.append(
                prediction
            )

            history.append(
                prediction
            )

        return np.asarray(
            predictions,
            dtype=float,
        )

    def _ensemble_forecast(
        self,
        series: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:

        models = {
            "naive": self._naive(
                series,
                horizon,
            ),
            "moving_average": self._moving_average(
                series,
                horizon,
            ),
            "exponential_smoothing":
                self._exponential_smoothing(
                    series,
                    horizon,
                ),
            "linear_trend":
                self._linear_trend_forecast(
                    series,
                    horizon,
                ),
            "autoregressive":
                self._autoregressive_forecast(
                    series,
                    horizon,
                ),
        }

        weights = {}

        for name in models:
            weights[name] = max(
                0.0,
                float(
                    self.config.ensemble_weights.get(
                        name,
                        0.0,
                    )
                ),
            )

        total_weight = sum(
            weights.values()
        )

        if total_weight <= 0:
            weights = {
                name: 1.0 / len(models)
                for name in models
            }
        else:
            weights = {
                name: value / total_weight
                for name, value in weights.items()
            }

        ensemble = np.zeros(
            horizon,
            dtype=float,
        )

        for name, predictions in models.items():
            ensemble += (
                weights[name]
                * predictions
            )

        return (
            ensemble,
            {
                "models": list(
                    models.keys()
                ),
                "weights": weights,
            },
        )

    # =================================================================
    # Prediction intervals
    # =================================================================

    def _estimate_residual_std(
        self,
        series: np.ndarray,
        method: ForecastMethod,
    ) -> float:

        if len(series) < 4:
            return float(
                np.std(series)
            )

        actual = series[1:]
        baseline = series[:-1]

        residuals = (
            actual - baseline
        )

        residual_std = float(
            np.std(
                residuals,
                ddof=1,
            )
        )

        if not np.isfinite(
            residual_std
        ):
            return 0.0

        return max(
            residual_std,
            1e-8,
        )

    def _prediction_intervals(
        self,
        predictions: np.ndarray,
        residual_std: float,
    ) -> tuple[np.ndarray, np.ndarray]:

        z = self._normal_z_value(
            self.config.confidence_level
        )

        # Uncertainty grows approximately with sqrt(horizon).
        steps = np.arange(
            1,
            len(predictions) + 1,
            dtype=float,
        )

        uncertainty = (
            z
            * residual_std
            * np.sqrt(steps)
        )

        return (
            predictions - uncertainty,
            predictions + uncertainty,
        )

    @staticmethod
    def _normal_z_value(
        confidence: float,
    ) -> float:

        # Common values avoid a scipy dependency.
        values = {
            0.80: 1.2816,
            0.90: 1.6449,
            0.95: 1.9600,
            0.98: 2.3263,
            0.99: 2.5758,
        }

        closest = min(
            values,
            key=lambda value: abs(
                value - confidence
            ),
        )

        return values[closest]

    # =================================================================
    # Trend
    # =================================================================

    def _trend_slope(
        self,
        series: np.ndarray,
    ) -> float:

        window = min(
            self.config.trend_window,
            len(series),
        )

        values = series[
            -window:
        ]

        if len(values) < 2:
            return 0.0

        x = np.arange(
            len(values),
            dtype=float,
        )

        slope = np.polyfit(
            x,
            values,
            1,
        )[0]

        return float(
            slope
        )

    @staticmethod
    def _classify_trend(
        slope: float,
        series: np.ndarray,
    ) -> ForecastTrend:

        baseline = max(
            abs(
                float(
                    np.mean(series)
                )
            ),
            1e-8,
        )

        normalized = (
            slope / baseline
        )

        if normalized >= 0.02:
            return ForecastTrend.RAPIDLY_INCREASING

        if normalized >= 0.005:
            return ForecastTrend.INCREASING

        if normalized <= -0.02:
            return ForecastTrend.RAPIDLY_DECREASING

        if normalized <= -0.005:
            return ForecastTrend.DECREASING

        return ForecastTrend.STABLE

    # =================================================================
    # Risk
    # =================================================================

    def _forecast_risk(
        self,
        value: float,
        threshold: Optional[ForecastThreshold],
    ) -> float:

        if threshold is None:
            return 0.0

        risks = []

        if threshold.healthy_max is not None:
            risks.append(
                self._upper_threshold_risk(
                    value,
                    threshold.healthy_max,
                    threshold.warning_max,
                    threshold.critical_max,
                )
            )

        if threshold.healthy_min is not None:
            risks.append(
                self._lower_threshold_risk(
                    value,
                    threshold.healthy_min,
                    threshold.warning_min,
                    threshold.critical_min,
                )
            )

        return self._clip01(
            max(
                risks,
                default=0.0,
            )
        )

    @staticmethod
    def _upper_threshold_risk(
        value: float,
        healthy: float,
        warning: Optional[float],
        critical: Optional[float],
    ) -> float:

        if value <= healthy:
            return 0.0

        warning = (
            float(warning)
            if warning is not None
            else healthy * 1.15
        )

        critical = (
            float(critical)
            if critical is not None
            else warning * 1.20
        )

        if value >= critical:
            return 1.0

        if value >= warning:
            span = max(
                critical - warning,
                1e-8,
            )

            return (
                0.70
                + 0.30
                * (
                    (value - warning)
                    / span
                )
            )

        span = max(
            warning - healthy,
            1e-8,
        )

        return (
            0.20
            + 0.50
            * (
                (value - healthy)
                / span
            )
        )

    @staticmethod
    def _lower_threshold_risk(
        value: float,
        healthy: float,
        warning: Optional[float],
        critical: Optional[float],
    ) -> float:

        if value >= healthy:
            return 0.0

        warning = (
            float(warning)
            if warning is not None
            else healthy * 0.85
        )

        critical = (
            float(critical)
            if critical is not None
            else warning * 0.75
        )

        if value <= critical:
            return 1.0

        if value <= warning:
            span = max(
                warning - critical,
                1e-8,
            )

            return (
                0.70
                + 0.30
                * (
                    (warning - value)
                    / span
                )
            )

        span = max(
            healthy - warning,
            1e-8,
        )

        return (
            0.20
            + 0.50
            * (
                (healthy - value)
                / span
            )
        )

    @staticmethod
    def _risk_level(
        score: float,
    ) -> ForecastRiskLevel:

        if score >= 0.85:
            return ForecastRiskLevel.CRITICAL

        if score >= 0.70:
            return ForecastRiskLevel.HIGH

        if score >= 0.45:
            return ForecastRiskLevel.MEDIUM

        if score >= 0.20:
            return ForecastRiskLevel.LOW

        return ForecastRiskLevel.NORMAL

    # =================================================================
    # Confidence
    # =================================================================

    def _confidence(
        self,
        *,
        series: np.ndarray,
        residual_std: float,
        predictions: np.ndarray,
    ) -> float:

        history_factor = min(
            1.0,
            len(series)
            / max(
                self.config.min_history * 4,
                1,
            ),
        )

        scale = max(
            float(
                np.std(series)
            ),
            abs(
                float(
                    np.mean(series)
                )
            )
            * 0.05,
            1e-8,
        )

        noise_ratio = (
            residual_std / scale
        )

        stability_factor = (
            1.0
            / (
                1.0
                + noise_ratio
            )
        )

        forecast_change = abs(
            float(predictions[-1])
            - float(series[-1])
        )

        extrapolation_ratio = (
            forecast_change
            / max(
                abs(float(series[-1])),
                scale,
                1e-8,
            )
        )

        extrapolation_factor = (
            1.0
            / (
                1.0
                + extrapolation_ratio
            )
        )

        confidence = (
            0.45 * history_factor
            + 0.35 * stability_factor
            + 0.20 * extrapolation_factor
        )

        return self._clip01(
            confidence
        )

    # =================================================================
    # Timestamp generation
    # =================================================================

    @staticmethod
    def _future_timestamps(
        *,
        timestamps: Optional[Sequence[Any]],
        horizon: int,
        sampling_interval: Optional[
            str | pd.Timedelta
        ],
    ) -> list[datetime]:

        if timestamps is None:
            return []

        parsed = pd.Series(
            pd.to_datetime(
                timestamps,
                errors="coerce",
                utc=True,
            )
        ).dropna()

        if parsed.empty:
            return []

        last_timestamp = parsed.iloc[-1]

        interval = sampling_interval

        if interval is None:
            if len(parsed) < 2:
                return []

            differences = (
                parsed
                .sort_values()
                .diff()
                .dropna()
            )

            differences = differences[
                differences > pd.Timedelta(0)
            ]

            if differences.empty:
                return []

            interval = differences.median()

        try:
            offset = (
                pd.tseries.frequencies.to_offset(
                    interval
                )
            )
        except (TypeError, ValueError):
            return []

        future = pd.date_range(
            start=last_timestamp + offset,
            periods=horizon,
            freq=offset,
        )

        return [
            timestamp.to_pydatetime()
            for timestamp in future
        ]

    # =================================================================
    # Helpers
    # =================================================================

    @staticmethod
    def _clean_values(
        values: Sequence[float]
        | np.ndarray
        | pd.Series,
    ) -> np.ndarray:

        array = np.asarray(
            pd.to_numeric(
                pd.Series(values),
                errors="coerce",
            ),
            dtype=float,
        )

        array = array[
            np.isfinite(array)
        ]

        if array.size == 0:
            raise ValueError(
                "No finite values available."
            )

        return array

    @staticmethod
    def _resolve_method(
        method: ForecastMethod | str,
    ) -> ForecastMethod:

        if isinstance(
            method,
            ForecastMethod,
        ):
            return method

        try:
            return ForecastMethod(
                str(method).lower()
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported forecasting method: {method}"
            ) from exc

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )


# =====================================================================
# Evaluation
# =====================================================================


@dataclass
class ForecastEvaluation:
    mae: float
    mse: float
    rmse: float
    mape: float
    smape: float
    bias: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mae": round(self.mae, 6),
            "mse": round(self.mse, 6),
            "rmse": round(self.rmse, 6),
            "mape": round(self.mape, 6),
            "smape": round(self.smape, 6),
            "bias": round(self.bias, 6),
            "r2": round(self.r2, 6),
        }


def evaluate_forecast(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> ForecastEvaluation:

    y_true = np.asarray(
        actual,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        predicted,
        dtype=float,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "actual and predicted must have identical shapes."
        )

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if not len(y_true):
        raise ValueError(
            "No finite observations available."
        )

    errors = (
        y_pred - y_true
    )

    absolute = np.abs(
        errors
    )

    mae = float(
        np.mean(absolute)
    )

    mse = float(
        np.mean(errors ** 2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    valid_mape = (
        np.abs(y_true) > epsilon
    )

    mape = (
        float(
            np.mean(
                absolute[valid_mape]
                / np.abs(
                    y_true[valid_mape]
                )
            )
            * 100.0
        )
        if valid_mape.any()
        else float("nan")
    )

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    valid_smape = (
        denominator > epsilon
    )

    smape = (
        float(
            np.mean(
                (
                    2.0
                    * absolute[valid_smape]
                )
                / denominator[valid_smape]
            )
            * 100.0
        )
        if valid_smape.any()
        else 0.0
    )

    bias = float(
        np.mean(errors)
    )

    denominator_r2 = float(
        np.sum(
            (
                y_true
                - np.mean(y_true)
            )
            ** 2
        )
    )

    r2 = (
        float(
            1.0
            - (
                np.sum(errors ** 2)
                / denominator_r2
            )
        )
        if denominator_r2 > epsilon
        else float("nan")
    )

    return ForecastEvaluation(
        mae=mae,
        mse=mse,
        rmse=rmse,
        mape=mape,
        smape=smape,
        bias=bias,
        r2=r2,
    )


# =====================================================================
# Default industrial model
# =====================================================================


def create_default_forecaster() -> IndustrialForecaster:

    return IndustrialForecaster(
        ForecastConfig(
            method=ForecastMethod.ENSEMBLE,
            horizon=24,
            moving_average_window=12,
            trend_window=24,
            exponential_alpha=0.30,
            autoregressive_lags=(
                1,
                2,
                3,
                6,
                12,
                24,
            ),
            confidence_level=0.95,
            thresholds={
                "temperature": ForecastThreshold(
                    healthy_max=70.0,
                    warning_max=85.0,
                    critical_max=110.0,
                ),
                "vibration_rms": ForecastThreshold(
                    healthy_max=4.5,
                    warning_max=7.0,
                    critical_max=12.0,
                ),
                "motor_current": ForecastThreshold(
                    healthy_max=30.0,
                    warning_max=40.0,
                    critical_max=55.0,
                ),
                "pressure": ForecastThreshold(
                    healthy_min=80.0,
                    healthy_max=120.0,
                    warning_min=65.0,
                    warning_max=140.0,
                    critical_min=45.0,
                    critical_max=165.0,
                ),
                "energy_consumption": ForecastThreshold(
                    healthy_max=100.0,
                    warning_max=125.0,
                    critical_max=165.0,
                ),
            },
        )
    )


# =====================================================================
# Convenience API
# =====================================================================


def forecast_signal(
    values: Sequence[float] | np.ndarray,
    *,
    signal_name: str,
    horizon: int = 24,
    timestamps: Optional[Sequence[Any]] = None,
) -> ForecastResult:

    model = create_default_forecaster()

    return model.forecast(
        values,
        signal_name=signal_name,
        horizon=horizon,
        timestamps=timestamps,
    )


# =====================================================================
# Demo
# =====================================================================


if __name__ == "__main__":

    rng = np.random.default_rng(
        42
    )

    observations = 240

    timestamps = pd.date_range(
        "2026-01-01",
        periods=observations,
        freq="h",
        tz="UTC",
    )

    temperature = (
        60.0
        + np.linspace(
            0,
            18,
            observations,
        )
        + 2.0
        * np.sin(
            np.arange(observations)
            * 2.0
            * np.pi
            / 24.0
        )
        + rng.normal(
            0,
            1.2,
            observations,
        )
    )

    forecaster = (
        create_default_forecaster()
    )

    result = forecaster.forecast(
        temperature,
        signal_name="temperature",
        horizon=24,
        timestamps=timestamps,
    )

    print(
        result.to_dict()
    )