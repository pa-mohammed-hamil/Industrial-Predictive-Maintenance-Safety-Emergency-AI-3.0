"""
Industrial Time-Series Feature Engineering
==========================================

Production-oriented feature engineering utilities for industrial telemetry.

Supports:
- lag features
- rolling statistics
- exponentially weighted statistics
- differences / rate of change
- percentage change
- trend / slope
- acceleration
- rolling RMS
- rolling energy
- coefficient of variation
- z-score
- threshold-distance features
- operating-hour features
- cyclical time features
- missing-data indicators
- cross-sensor interaction features
- health/degradation features
- leakage-safe historical transformations
- feature sanitization

Designed for:
    forecasting.py
    sequence_models.py
    transformer_models.py
    degradation_model.py
    failure_prediction.py
    rul_prediction.py
    maintenance_forecasting.py
    anomaly_detection/*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class SensorThreshold:
    """
    Optional engineering limits for one sensor.

    Example:
        temperature:
            healthy_min=20
            healthy_max=70
            warning_min=10
            warning_max=85
            critical_min=0
            critical_max=110
    """

    healthy_min: Optional[float] = None
    healthy_max: Optional[float] = None

    warning_min: Optional[float] = None
    warning_max: Optional[float] = None

    critical_min: Optional[float] = None
    critical_max: Optional[float] = None


@dataclass
class FeatureEngineeringConfig:
    timestamp_column: str = "timestamp"

    sensor_columns: Optional[Sequence[str]] = None

    lag_steps: Sequence[int] = (
        1,
        2,
        3,
        6,
        12,
        24,
    )

    rolling_windows: Sequence[int] = (
        3,
        6,
        12,
        24,
    )

    ewm_spans: Sequence[int] = (
        6,
        12,
        24,
    )

    trend_windows: Sequence[int] = (
        6,
        12,
        24,
    )

    add_time_features: bool = True
    add_cyclical_time_features: bool = True

    add_lag_features: bool = True
    add_rolling_features: bool = True
    add_ewm_features: bool = True

    add_difference_features: bool = True
    add_rate_features: bool = True
    add_trend_features: bool = True

    add_rms_features: bool = True
    add_energy_features: bool = True

    add_zscore_features: bool = True
    add_missing_features: bool = True

    add_cross_sensor_features: bool = True

    # shift(1) before rolling operations prevents the current observation
    # from entering historical rolling features.
    leakage_safe: bool = True

    fill_infinite: bool = True

    # Optional final fill. None preserves NaN values created by lags/windows.
    fillna_value: Optional[float] = None

    thresholds: Mapping[str, SensorThreshold] = field(
        default_factory=dict
    )


# =====================================================================
# Feature Engineer
# =====================================================================


class TimeSeriesFeatureEngineer:
    """
    Reusable feature-engineering pipeline for industrial telemetry.

    The class intentionally does not fit global scalers or learn statistics
    from the full dataset. Scaling should be performed in a training pipeline
    after train/validation/test splitting to avoid data leakage.
    """

    def __init__(
        self,
        config: Optional[FeatureEngineeringConfig] = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else FeatureEngineeringConfig()
        )

    # =================================================================
    # Main API
    # =================================================================

    def transform(
        self,
        dataframe: pd.DataFrame,
        *,
        sensor_columns: Optional[Sequence[str]] = None,
        copy: bool = True,
    ) -> pd.DataFrame:

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            return dataframe.copy() if copy else dataframe

        df = dataframe.copy() if copy else dataframe

        timestamp_column = self.config.timestamp_column

        # -------------------------------------------------------------
        # Timestamp normalization
        # -------------------------------------------------------------

        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(
                df[timestamp_column],
                errors="coerce",
                utc=True,
            )

            df = df.sort_values(
                timestamp_column,
                kind="stable",
            ).reset_index(drop=True)

        # -------------------------------------------------------------
        # Resolve sensors
        # -------------------------------------------------------------

        sensors = self._resolve_sensor_columns(
            df,
            sensor_columns,
        )

        # Force selected sensor columns to numeric.
        for column in sensors:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # -------------------------------------------------------------
        # Missingness BEFORE filling/transforming
        # -------------------------------------------------------------

        if self.config.add_missing_features:
            df = self._add_missing_features(
                df,
                sensors,
            )

        # -------------------------------------------------------------
        # Time features
        # -------------------------------------------------------------

        if (
            self.config.add_time_features
            and timestamp_column in df.columns
        ):
            df = self._add_time_features(
                df
            )

        # -------------------------------------------------------------
        # Per-sensor features
        # -------------------------------------------------------------

        for sensor in sensors:

            if self.config.add_lag_features:
                df = self._add_lag_features(
                    df,
                    sensor,
                )

            if self.config.add_difference_features:
                df = self._add_difference_features(
                    df,
                    sensor,
                )

            if self.config.add_rate_features:
                df = self._add_rate_features(
                    df,
                    sensor,
                )

            if self.config.add_rolling_features:
                df = self._add_rolling_features(
                    df,
                    sensor,
                )

            if self.config.add_ewm_features:
                df = self._add_ewm_features(
                    df,
                    sensor,
                )

            if self.config.add_trend_features:
                df = self._add_trend_features(
                    df,
                    sensor,
                )

            if self.config.add_rms_features:
                df = self._add_rms_features(
                    df,
                    sensor,
                )

            if self.config.add_energy_features:
                df = self._add_energy_features(
                    df,
                    sensor,
                )

            if self.config.add_zscore_features:
                df = self._add_zscore_features(
                    df,
                    sensor,
                )

            df = self._add_threshold_features(
                df,
                sensor,
            )

        # -------------------------------------------------------------
        # Cross-sensor interactions
        # -------------------------------------------------------------

        if (
            self.config.add_cross_sensor_features
            and len(sensors) >= 2
        ):
            df = self._add_cross_sensor_features(
                df,
                sensors,
            )

        # -------------------------------------------------------------
        # Sanitize
        # -------------------------------------------------------------

        if self.config.fill_infinite:
            df = df.replace(
                [np.inf, -np.inf],
                np.nan,
            )

        if self.config.fillna_value is not None:
            numeric_columns = df.select_dtypes(
                include=[np.number]
            ).columns

            df[numeric_columns] = (
                df[numeric_columns]
                .fillna(
                    float(
                        self.config.fillna_value
                    )
                )
            )

        return df

    # =================================================================
    # Sensor resolution
    # =================================================================

    def _resolve_sensor_columns(
        self,
        df: pd.DataFrame,
        sensor_columns: Optional[Sequence[str]],
    ) -> list[str]:

        requested = (
            sensor_columns
            or self.config.sensor_columns
        )

        if requested is not None:

            missing = [
                column
                for column in requested
                if column not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Sensor columns not found: {missing}"
                )

            return list(
                dict.fromkeys(requested)
            )

        excluded = {
            self.config.timestamp_column,
            "asset_id",
            "machine_id",
            "component_id",
            "site_id",
            "zone_id",
            "label",
            "target",
            "failure",
            "failure_label",
            "rul",
            "rul_hours",
        }

        return [
            column
            for column in df.select_dtypes(
                include=[np.number]
            ).columns
            if column not in excluded
        ]

    # =================================================================
    # Historical series
    # =================================================================

    def _history(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """
        Return the series used by historical features.

        In leakage-safe mode, the current row is excluded.
        """

        if self.config.leakage_safe:
            return series.shift(1)

        return series

    # =================================================================
    # Time features
    # =================================================================

    def _add_time_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        column = self.config.timestamp_column
        timestamp = df[column]

        df["time_hour"] = timestamp.dt.hour
        df["time_day_of_week"] = timestamp.dt.dayofweek
        df["time_day_of_month"] = timestamp.dt.day
        df["time_month"] = timestamp.dt.month
        df["time_quarter"] = timestamp.dt.quarter
        df["time_day_of_year"] = timestamp.dt.dayofyear

        df["time_is_weekend"] = (
            timestamp.dt.dayofweek >= 5
        ).astype("int8")

        if self.config.add_cyclical_time_features:

            hour = timestamp.dt.hour.astype(float)

            df["time_hour_sin"] = np.sin(
                2.0 * np.pi * hour / 24.0
            )

            df["time_hour_cos"] = np.cos(
                2.0 * np.pi * hour / 24.0
            )

            dow = timestamp.dt.dayofweek.astype(float)

            df["time_dow_sin"] = np.sin(
                2.0 * np.pi * dow / 7.0
            )

            df["time_dow_cos"] = np.cos(
                2.0 * np.pi * dow / 7.0
            )

            month = timestamp.dt.month.astype(float)

            df["time_month_sin"] = np.sin(
                2.0 * np.pi * month / 12.0
            )

            df["time_month_cos"] = np.cos(
                2.0 * np.pi * month / 12.0
            )

        return df

    # =================================================================
    # Missing-data features
    # =================================================================

    @staticmethod
    def _add_missing_features(
        df: pd.DataFrame,
        sensors: Sequence[str],
    ) -> pd.DataFrame:

        for sensor in sensors:
            df[f"{sensor}_missing"] = (
                df[sensor]
                .isna()
                .astype("int8")
            )

        df["sensor_missing_count"] = (
            df[list(sensors)]
            .isna()
            .sum(axis=1)
        )

        df["sensor_missing_ratio"] = (
            df["sensor_missing_count"]
            / max(
                len(sensors),
                1,
            )
        )

        return df

    # =================================================================
    # Lag features
    # =================================================================

    def _add_lag_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        for lag in self.config.lag_steps:

            if lag <= 0:
                continue

            df[
                f"{sensor}_lag_{lag}"
            ] = df[sensor].shift(
                int(lag)
            )

        return df

    # =================================================================
    # Difference features
    # =================================================================

    def _add_difference_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        series = df[sensor]

        df[
            f"{sensor}_diff_1"
        ] = series.diff(1)

        df[
            f"{sensor}_diff_2"
        ] = series.diff(2)

        df[
            f"{sensor}_pct_change_1"
        ] = (
            series
            .pct_change(
                periods=1,
                fill_method=None,
            )
        )

        return df

    # =================================================================
    # Rate / acceleration
    # =================================================================

    def _add_rate_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        series = df[sensor]

        timestamp_column = (
            self.config.timestamp_column
        )

        if timestamp_column in df.columns:

            delta_seconds = (
                df[timestamp_column]
                .diff()
                .dt.total_seconds()
            )

            delta_hours = (
                delta_seconds / 3600.0
            )

            delta_hours = delta_hours.where(
                delta_hours > 0
            )

            rate = (
                series.diff()
                / delta_hours
            )

            df[
                f"{sensor}_rate_per_hour"
            ] = rate

            df[
                f"{sensor}_acceleration_per_hour"
            ] = (
                rate.diff()
                / delta_hours
            )

        else:

            rate = series.diff()

            df[
                f"{sensor}_rate"
            ] = rate

            df[
                f"{sensor}_acceleration"
            ] = rate.diff()

        return df

    # =================================================================
    # Rolling features
    # =================================================================

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        history = self._history(
            df[sensor]
        )

        for window in self.config.rolling_windows:

            window = int(window)

            if window <= 1:
                continue

            minimum = max(
                2,
                min(
                    window,
                    window // 2,
                ),
            )

            rolling = history.rolling(
                window=window,
                min_periods=minimum,
            )

            prefix = (
                f"{sensor}_roll_{window}"
            )

            df[
                f"{prefix}_mean"
            ] = rolling.mean()

            df[
                f"{prefix}_std"
            ] = rolling.std(
                ddof=0
            )

            df[
                f"{prefix}_min"
            ] = rolling.min()

            df[
                f"{prefix}_max"
            ] = rolling.max()

            df[
                f"{prefix}_median"
            ] = rolling.median()

            df[
                f"{prefix}_range"
            ] = (
                df[f"{prefix}_max"]
                - df[f"{prefix}_min"]
            )

            denominator = (
                df[f"{prefix}_mean"]
                .abs()
                .replace(
                    0.0,
                    np.nan,
                )
            )

            df[
                f"{prefix}_cv"
            ] = (
                df[f"{prefix}_std"]
                / denominator
            )

        return df

    # =================================================================
    # EWM features
    # =================================================================

    def _add_ewm_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        history = self._history(
            df[sensor]
        )

        for span in self.config.ewm_spans:

            span = int(span)

            if span <= 1:
                continue

            ewm = history.ewm(
                span=span,
                adjust=False,
                min_periods=2,
            )

            df[
                f"{sensor}_ewm_{span}_mean"
            ] = ewm.mean()

            df[
                f"{sensor}_ewm_{span}_std"
            ] = ewm.std(
                bias=True
            )

        return df

    # =================================================================
    # Trend / slope
    # =================================================================

    def _add_trend_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        history = self._history(
            df[sensor]
        )

        for window in self.config.trend_windows:

            window = int(window)

            if window < 2:
                continue

            df[
                f"{sensor}_trend_{window}"
            ] = (
                history
                .rolling(
                    window=window,
                    min_periods=max(
                        2,
                        window // 2,
                    ),
                )
                .apply(
                    self._linear_slope,
                    raw=True,
                )
            )

        return df

    # =================================================================
    # RMS
    # =================================================================

    def _add_rms_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        history = self._history(
            df[sensor]
        )

        for window in self.config.rolling_windows:

            window = int(window)

            if window <= 1:
                continue

            df[
                f"{sensor}_rms_{window}"
            ] = (
                history
                .pow(2)
                .rolling(
                    window=window,
                    min_periods=max(
                        2,
                        window // 2,
                    ),
                )
                .mean()
                .pow(0.5)
            )

        return df

    # =================================================================
    # Energy
    # =================================================================

    def _add_energy_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        history = self._history(
            df[sensor]
        )

        squared = history.pow(
            2
        )

        for window in self.config.rolling_windows:

            window = int(window)

            if window <= 1:
                continue

            df[
                f"{sensor}_energy_{window}"
            ] = (
                squared
                .rolling(
                    window=window,
                    min_periods=max(
                        2,
                        window // 2,
                    ),
                )
                .sum()
            )

        return df

    # =================================================================
    # Rolling z-score
    # =================================================================

    def _add_zscore_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        history = self._history(
            df[sensor]
        )

        for window in self.config.rolling_windows:

            window = int(window)

            if window <= 2:
                continue

            rolling = history.rolling(
                window=window,
                min_periods=max(
                    3,
                    window // 2,
                ),
            )

            mean = rolling.mean()

            std = (
                rolling.std(
                    ddof=0
                )
                .replace(
                    0.0,
                    np.nan,
                )
            )

            # Compare the current observation with historical behavior.
            df[
                f"{sensor}_zscore_{window}"
            ] = (
                df[sensor] - mean
            ) / std

        return df

    # =================================================================
    # Engineering-threshold features
    # =================================================================

    def _add_threshold_features(
        self,
        df: pd.DataFrame,
        sensor: str,
    ) -> pd.DataFrame:

        threshold = (
            self.config.thresholds.get(
                sensor
            )
        )

        if threshold is None:
            return df

        values = df[
            sensor
        ]

        if threshold.healthy_max is not None:

            healthy_max = float(
                threshold.healthy_max
            )

            scale = max(
                abs(healthy_max),
                1e-9,
            )

            df[
                f"{sensor}_above_healthy"
            ] = np.maximum(
                values - healthy_max,
                0.0,
            )

            df[
                f"{sensor}_above_healthy_ratio"
            ] = (
                df[f"{sensor}_above_healthy"]
                / scale
            )

        if threshold.healthy_min is not None:

            healthy_min = float(
                threshold.healthy_min
            )

            scale = max(
                abs(healthy_min),
                1.0,
            )

            df[
                f"{sensor}_below_healthy"
            ] = np.maximum(
                healthy_min - values,
                0.0,
            )

            df[
                f"{sensor}_below_healthy_ratio"
            ] = (
                df[f"{sensor}_below_healthy"]
                / scale
            )

        if threshold.warning_max is not None:

            df[
                f"{sensor}_warning_high"
            ] = (
                values
                >= float(
                    threshold.warning_max
                )
            ).astype(
                "int8"
            )

        if threshold.warning_min is not None:

            df[
                f"{sensor}_warning_low"
            ] = (
                values
                <= float(
                    threshold.warning_min
                )
            ).astype(
                "int8"
            )

        if threshold.critical_max is not None:

            df[
                f"{sensor}_critical_high"
            ] = (
                values
                >= float(
                    threshold.critical_max
                )
            ).astype(
                "int8"
            )

        if threshold.critical_min is not None:

            df[
                f"{sensor}_critical_low"
            ] = (
                values
                <= float(
                    threshold.critical_min
                )
            ).astype(
                "int8"
            )

        return df

    # =================================================================
    # Cross-sensor features
    # =================================================================

    @staticmethod
    def _add_cross_sensor_features(
        df: pd.DataFrame,
        sensors: Sequence[str],
    ) -> pd.DataFrame:

        sensor_frame = df[
            list(sensors)
        ]

        df[
            "sensor_cross_mean"
        ] = sensor_frame.mean(
            axis=1
        )

        df[
            "sensor_cross_std"
        ] = sensor_frame.std(
            axis=1,
            ddof=0,
        )

        df[
            "sensor_cross_min"
        ] = sensor_frame.min(
            axis=1
        )

        df[
            "sensor_cross_max"
        ] = sensor_frame.max(
            axis=1
        )

        df[
            "sensor_cross_range"
        ] = (
            df["sensor_cross_max"]
            - df["sensor_cross_min"]
        )

        df[
            "sensor_cross_energy"
        ] = (
            sensor_frame
            .pow(2)
            .sum(axis=1)
        )

        return df

    # =================================================================
    # Helpers
    # =================================================================

    @staticmethod
    def _linear_slope(
        values: np.ndarray,
    ) -> float:

        values = np.asarray(
            values,
            dtype=float,
        )

        valid = np.isfinite(
            values
        )

        if valid.sum() < 2:
            return np.nan

        y = values[
            valid
        ]

        x = np.arange(
            len(values),
            dtype=float,
        )[valid]

        x_mean = x.mean()
        y_mean = y.mean()

        denominator = np.sum(
            (x - x_mean) ** 2
        )

        if denominator <= 0:
            return 0.0

        numerator = np.sum(
            (x - x_mean)
            * (y - y_mean)
        )

        return float(
            numerator
            / denominator
        )


# =====================================================================
# Domain-specific helper features
# =====================================================================


def add_operating_features(
    dataframe: pd.DataFrame,
    *,
    operating_hours_column: str = "operating_hours",
    maintenance_hours_column: Optional[str] = "hours_since_maintenance",
    rated_hours: Optional[float] = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add lifecycle / operating-hour features.
    """

    df = (
        dataframe.copy()
        if copy
        else dataframe
    )

    if operating_hours_column in df.columns:

        operating_hours = pd.to_numeric(
            df[operating_hours_column],
            errors="coerce",
        ).clip(
            lower=0
        )

        df[
            "operating_hours_log1p"
        ] = np.log1p(
            operating_hours
        )

        if (
            rated_hours is not None
            and rated_hours > 0
        ):

            df[
                "operating_life_ratio"
            ] = (
                operating_hours
                / float(rated_hours)
            )

    if (
        maintenance_hours_column
        and maintenance_hours_column in df.columns
    ):

        maintenance_age = pd.to_numeric(
            df[maintenance_hours_column],
            errors="coerce",
        ).clip(
            lower=0
        )

        df[
            "maintenance_age_log1p"
        ] = np.log1p(
            maintenance_age
        )

    return df


def add_health_features(
    dataframe: pd.DataFrame,
    *,
    degradation_column: str = "degradation_score",
    anomaly_column: str = "anomaly_score",
    failure_probability_column: str = "failure_probability",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add higher-level predictive-maintenance interaction features.
    """

    df = (
        dataframe.copy()
        if copy
        else dataframe
    )

    available = []

    for column in (
        degradation_column,
        anomaly_column,
        failure_probability_column,
    ):

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .clip(
                    lower=0.0,
                    upper=1.0,
                )
            )

            available.append(
                column
            )

    if not available:
        return df

    frame = df[
        available
    ]

    df[
        "predictive_risk_mean"
    ] = frame.mean(
        axis=1
    )

    df[
        "predictive_risk_max"
    ] = frame.max(
        axis=1
    )

    if (
        degradation_column in df.columns
        and anomaly_column in df.columns
    ):

        df[
            "degradation_anomaly_interaction"
        ] = (
            df[degradation_column]
            * df[anomaly_column]
        )

    if (
        degradation_column in df.columns
        and failure_probability_column in df.columns
    ):

        df[
            "degradation_failure_interaction"
        ] = (
            df[degradation_column]
            * df[
                failure_probability_column
            ]
        )

    if (
        anomaly_column in df.columns
        and failure_probability_column in df.columns
    ):

        df[
            "anomaly_failure_interaction"
        ] = (
            df[anomaly_column]
            * df[
                failure_probability_column
            ]
        )

    return df


# =====================================================================
# Sequence builder
# =====================================================================


def create_sequences(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: Optional[str] = None,
    sequence_length: int = 24,
    horizon: int = 1,
    drop_nan: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert tabular time-series data into model-ready sequences.

    X shape:
        (samples, sequence_length, features)

    y shape:
        (samples,)

    Useful for:
        LSTM
        GRU
        Temporal CNN
        Transformer
        RUL models
        forecasting models
    """

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Feature columns not found: {missing}"
        )

    if (
        target_column is not None
        and target_column not in dataframe.columns
    ):
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    features = (
        dataframe[
            list(feature_columns)
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    targets = None

    if target_column is not None:
        targets = pd.to_numeric(
            dataframe[target_column],
            errors="coerce",
        ).to_numpy(
            dtype=np.float32
        )

    x_sequences: list[np.ndarray] = []
    y_values: list[float] = []

    final_start = (
        len(dataframe)
        - sequence_length
        - horizon
        + 1
    )

    for start in range(
        max(
            final_start,
            0,
        )
    ):

        end = (
            start
            + sequence_length
        )

        target_index = (
            end
            + horizon
            - 1
        )

        sequence = features[
            start:end
        ]

        if (
            drop_nan
            and not np.isfinite(
                sequence
            ).all()
        ):
            continue

        if targets is not None:

            target = targets[
                target_index
            ]

            if (
                drop_nan
                and not np.isfinite(
                    target
                )
            ):
                continue

            y_values.append(
                float(target)
            )

        x_sequences.append(
            sequence
        )

    if x_sequences:

        x = np.stack(
            x_sequences
        ).astype(
            np.float32
        )

    else:

        x = np.empty(
            (
                0,
                sequence_length,
                len(feature_columns),
            ),
            dtype=np.float32,
        )

    if targets is None:
        return x, None

    y = np.asarray(
        y_values,
        dtype=np.float32,
    )

    return x, y


# =====================================================================
# Feature-name helper
# =====================================================================


def get_model_feature_columns(
    dataframe: pd.DataFrame,
    *,
    exclude: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Return numeric columns suitable for model input.
    """

    excluded = set(
        exclude or ()
    )

    excluded.update(
        {
            "target",
            "label",
            "failure_label",
        }
    )

    return [
        column
        for column in dataframe.select_dtypes(
            include=[np.number]
        ).columns
        if column not in excluded
    ]


# =====================================================================
# Default industrial configuration
# =====================================================================


def create_default_feature_engineer(
) -> TimeSeriesFeatureEngineer:

    config = FeatureEngineeringConfig(
        lag_steps=(
            1,
            2,
            3,
            6,
            12,
            24,
        ),

        rolling_windows=(
            3,
            6,
            12,
            24,
        ),

        ewm_spans=(
            6,
            12,
            24,
        ),

        trend_windows=(
            6,
            12,
            24,
        ),

        leakage_safe=True,

        thresholds={
            "temperature": SensorThreshold(
                healthy_max=70.0,
                warning_max=85.0,
                critical_max=110.0,
            ),

            "vibration_rms": SensorThreshold(
                healthy_max=4.5,
                warning_max=7.0,
                critical_max=12.0,
            ),

            "motor_current": SensorThreshold(
                healthy_max=30.0,
                warning_max=40.0,
                critical_max=55.0,
            ),

            "pressure": SensorThreshold(
                healthy_min=80.0,
                healthy_max=120.0,

                warning_min=65.0,
                warning_max=140.0,

                critical_min=45.0,
                critical_max=165.0,
            ),

            "energy_consumption": SensorThreshold(
                healthy_max=100.0,
                warning_max=125.0,
                critical_max=165.0,
            ),
        },
    )

    return TimeSeriesFeatureEngineer(
        config
    )


# =====================================================================
# Convenience function
# =====================================================================


def engineer_features(
    dataframe: pd.DataFrame,
    *,
    sensor_columns: Optional[
        Sequence[str]
    ] = None,
) -> pd.DataFrame:

    engineer = (
        create_default_feature_engineer()
    )

    return engineer.transform(
        dataframe,
        sensor_columns=sensor_columns,
    )


# =====================================================================
# Demo
# =====================================================================


if __name__ == "__main__":

    rng = np.random.default_rng(
        42
    )

    rows = 100

    demo = pd.DataFrame(
        {
            "timestamp":
                pd.date_range(
                    "2026-01-01",
                    periods=rows,
                    freq="h",
                    tz="UTC",
                ),

            "temperature":
                60
                + np.linspace(
                    0,
                    20,
                    rows,
                )
                + rng.normal(
                    0,
                    2,
                    rows,
                ),

            "vibration_rms":
                3.5
                + np.linspace(
                    0,
                    4,
                    rows,
                )
                + rng.normal(
                    0,
                    0.3,
                    rows,
                ),

            "motor_current":
                24
                + rng.normal(
                    0,
                    2,
                    rows,
                ),

            "pressure":
                100
                + rng.normal(
                    0,
                    5,
                    rows,
                ),

            "energy_consumption":
                85
                + np.linspace(
                    0,
                    25,
                    rows,
                )
                + rng.normal(
                    0,
                    3,
                    rows,
                ),
        }
    )

    engineer = (
        create_default_feature_engineer()
    )

    features = engineer.transform(
        demo,
        sensor_columns=[
            "temperature",
            "vibration_rms",
            "motor_current",
            "pressure",
            "energy_consumption",
        ],
    )

    print(
        "Input shape:",
        demo.shape,
    )

    print(
        "Feature shape:",
        features.shape,
    )

    print(
        "\nGenerated features:"
    )

    for feature in features.columns:
        print(
            feature
        )