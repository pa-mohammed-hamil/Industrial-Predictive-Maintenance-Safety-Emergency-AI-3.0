"""
Industrial Time-Series Forecasting Utilities
============================================

Shared utilities for industrial forecasting and predictive-maintenance
time-series models.

Features:
- chronological train/validation/test splitting
- sliding-window sequence generation
- multi-step forecasting targets
- lag-aligned supervised datasets
- leakage-safe numeric scaling
- inverse scaling
- missing-value handling
- timestamp validation
- frequency inference
- future timestamp generation
- forecast metrics
- interval / residual utilities
- rolling backtesting splits
- naive baselines
- prediction alignment
- deterministic reproducibility helpers

Designed for:
    forecasting.py
    feature_engineering.py
    sequence_models.py
    transformer_models.py
    maintenance_forecasting.py
    rul_prediction.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import pandas as pd


# =====================================================================
# Data containers
# =====================================================================


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class SequenceDataset:
    X: np.ndarray
    y: Optional[np.ndarray]
    target_indices: np.ndarray
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    sequence_length: int
    horizon: int


@dataclass(frozen=True)
class BacktestSplit:
    fold: int

    train_start: int
    train_end: int

    validation_start: int
    validation_end: int

    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass
class ForecastMetrics:
    mae: float
    mse: float
    rmse: float
    mape: float
    smape: float
    wape: float
    bias: float
    r2: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mae": round(self.mae, 6),
            "mse": round(self.mse, 6),
            "rmse": round(self.rmse, 6),
            "mape": round(self.mape, 6),
            "smape": round(self.smape, 6),
            "wape": round(self.wape, 6),
            "bias": round(self.bias, 6),
            "r2": round(self.r2, 6),
        }


# =====================================================================
# Leakage-safe scaler
# =====================================================================


class StandardScaler:
    """
    Lightweight NumPy/Pandas standard scaler.

    IMPORTANT:
    Fit this scaler on the training split only.

    Do not fit using validation/test data.
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-8,
    ) -> None:

        self.epsilon = float(epsilon)

        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

        self.feature_names_: Optional[
            tuple[str, ...]
        ] = None

        self.fitted_: bool = False

    def fit(
        self,
        values: np.ndarray | pd.DataFrame,
    ) -> "StandardScaler":

        array, names = self._to_array(
            values
        )

        if array.ndim != 2:
            raise ValueError(
                "StandardScaler expects 2D input."
            )

        self.mean_ = np.nanmean(
            array,
            axis=0,
        )

        scale = np.nanstd(
            array,
            axis=0,
        )

        scale = np.where(
            np.isfinite(scale)
            & (scale > self.epsilon),
            scale,
            1.0,
        )

        self.scale_ = scale

        self.feature_names_ = names
        self.fitted_ = True

        return self

    def transform(
        self,
        values: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:

        self._check_fitted()

        array, _ = self._to_array(
            values
        )

        return (
            array - self.mean_
        ) / self.scale_

    def fit_transform(
        self,
        values: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:

        return self.fit(
            values
        ).transform(
            values
        )

    def inverse_transform(
        self,
        values: np.ndarray,
    ) -> np.ndarray:

        self._check_fitted()

        array = np.asarray(
            values,
            dtype=float,
        )

        return (
            array * self.scale_
            + self.mean_
        )

    def state_dict(
        self,
    ) -> Dict[str, Any]:

        self._check_fitted()

        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "feature_names": (
                list(self.feature_names_)
                if self.feature_names_
                is not None
                else None
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, Any],
    ) -> "StandardScaler":

        scaler = cls()

        scaler.mean_ = np.asarray(
            state["mean"],
            dtype=float,
        )

        scaler.scale_ = np.asarray(
            state["scale"],
            dtype=float,
        )

        names = state.get(
            "feature_names"
        )

        scaler.feature_names_ = (
            tuple(names)
            if names is not None
            else None
        )

        scaler.fitted_ = True

        return scaler

    def _check_fitted(
        self,
    ) -> None:

        if (
            not self.fitted_
            or self.mean_ is None
            or self.scale_ is None
        ):
            raise RuntimeError(
                "Scaler has not been fitted."
            )

    @staticmethod
    def _to_array(
        values: np.ndarray | pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        Optional[tuple[str, ...]],
    ]:

        if isinstance(
            values,
            pd.DataFrame,
        ):

            return (
                values.to_numpy(
                    dtype=float
                ),
                tuple(
                    values.columns
                ),
            )

        return (
            np.asarray(
                values,
                dtype=float,
            ),
            None,
        )


# =====================================================================
# Chronological split
# =====================================================================


def chronological_split(
    dataframe: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_column: Optional[str] = "timestamp",
    reset_index: bool = True,
) -> TimeSeriesSplit:
    """
    Chronological split.

    Never randomly shuffle forecasting datasets.
    """

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if not np.isclose(
        total_ratio,
        1.0,
    ):
        raise ValueError(
            "train_ratio + validation_ratio + "
            "test_ratio must equal 1."
        )

    if min(
        train_ratio,
        validation_ratio,
        test_ratio,
    ) < 0:
        raise ValueError(
            "Split ratios cannot be negative."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot split an empty dataframe."
        )

    df = dataframe.copy()

    if (
        timestamp_column
        and timestamp_column in df.columns
    ):

        df[timestamp_column] = pd.to_datetime(
            df[timestamp_column],
            errors="coerce",
            utc=True,
        )

        df = df.sort_values(
            timestamp_column,
            kind="stable",
        )

    n = len(
        df
    )

    train_end = int(
        n * train_ratio
    )

    validation_end = (
        train_end
        + int(
            n * validation_ratio
        )
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    if reset_index:

        train.reset_index(
            drop=True,
            inplace=True,
        )

        validation.reset_index(
            drop=True,
            inplace=True,
        )

        test.reset_index(
            drop=True,
            inplace=True,
        )

    return TimeSeriesSplit(
        train=train,
        validation=validation,
        test=test,
    )


# =====================================================================
# Timestamp validation
# =====================================================================


def prepare_time_series(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    drop_invalid_timestamps: bool = True,
    drop_duplicate_timestamps: bool = False,
) -> pd.DataFrame:

    if timestamp_column not in dataframe.columns:
        raise ValueError(
            f"Missing timestamp column: "
            f"{timestamp_column}"
        )

    df = dataframe.copy()

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )

    if drop_invalid_timestamps:

        df = df.dropna(
            subset=[
                timestamp_column
            ]
        )

    if drop_duplicate_timestamps:

        df = df.drop_duplicates(
            subset=[
                timestamp_column
            ],
            keep="last",
        )

    df = (
        df.sort_values(
            timestamp_column,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# =====================================================================
# Frequency
# =====================================================================


def infer_frequency(
    timestamps: Sequence[Any] | pd.Series,
) -> Optional[pd.Timedelta]:

    series = pd.Series(
        pd.to_datetime(
            timestamps,
            errors="coerce",
            utc=True,
        )
    ).dropna()

    if len(series) < 2:
        return None

    series = (
        series
        .sort_values()
        .drop_duplicates()
    )

    differences = (
        series
        .diff()
        .dropna()
    )

    differences = differences[
        differences > pd.Timedelta(0)
    ]

    if differences.empty:
        return None

    return differences.median()


def generate_future_timestamps(
    last_timestamp: Any,
    *,
    periods: int,
    frequency: str | pd.Timedelta,
) -> pd.DatetimeIndex:

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    timestamp = pd.to_datetime(
        last_timestamp,
        utc=True,
    )

    offset = pd.tseries.frequencies.to_offset(
        frequency
    )

    return pd.date_range(
        start=timestamp + offset,
        periods=periods,
        freq=offset,
    )


# =====================================================================
# Missing-value handling
# =====================================================================


def fill_time_series_missing(
    dataframe: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    method: str = "interpolate",
    limit: Optional[int] = None,
) -> pd.DataFrame:

    df = dataframe.copy()

    if columns is None:

        columns = list(
            df.select_dtypes(
                include=[np.number]
            ).columns
        )

    for column in columns:

        if column not in df.columns:
            raise ValueError(
                f"Column not found: {column}"
            )

        if method == "interpolate":

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .interpolate(
                    method="linear",
                    limit=limit,
                    limit_direction="both",
                )
            )

        elif method == "forward":

            df[column] = (
                df[column]
                .ffill(
                    limit=limit
                )
            )

        elif method == "backward":

            df[column] = (
                df[column]
                .bfill(
                    limit=limit
                )
            )

        elif method == "zero":

            df[column] = (
                df[column]
                .fillna(
                    0.0
                )
            )

        elif method == "median":

            median = pd.to_numeric(
                df[column],
                errors="coerce",
            ).median()

            df[column] = (
                df[column]
                .fillna(
                    median
                )
            )

        else:
            raise ValueError(
                "method must be one of: "
                "interpolate, forward, backward, "
                "zero, median"
            )

    return df


# =====================================================================
# Sequence generation
# =====================================================================


def create_forecasting_sequences(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Optional[
        Sequence[str]
    ] = None,
    sequence_length: int = 24,
    horizon: int = 1,
    multi_step: bool = False,
    drop_nan: bool = True,
    dtype: np.dtype = np.float32,
) -> SequenceDataset:
    """
    Build sequence-model datasets.

    multi_step=False:
        X[t-seq:t] -> y[t+horizon-1]

    multi_step=True:
        X[t-seq:t] -> y[t:t+horizon]

    Shapes
    ------
    Single-step:
        X: [N, sequence_length, F]
        y: [N, T]

    Multi-step:
        X: [N, sequence_length, F]
        y: [N, horizon, T]
    """

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    features = tuple(
        feature_columns
    )

    targets = tuple(
        target_columns or ()
    )

    missing_features = [
        column
        for column in features
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: "
            f"{missing_features}"
        )

    missing_targets = [
        column
        for column in targets
        if column not in dataframe.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: "
            f"{missing_targets}"
        )

    feature_array = (
        dataframe[
            list(features)
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=dtype,
        )
    )

    target_array: Optional[
        np.ndarray
    ] = None

    if targets:

        target_array = (
            dataframe[
                list(targets)
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=dtype,
            )
        )

    X_values: list[
        np.ndarray
    ] = []

    y_values: list[
        np.ndarray
    ] = []

    target_indices: list[
        int
    ] = []

    n = len(
        dataframe
    )

    last_start = (
        n
        - sequence_length
        - horizon
        + 1
    )

    for start in range(
        max(
            0,
            last_start,
        )
    ):

        sequence_end = (
            start
            + sequence_length
        )

        X_window = feature_array[
            start:sequence_end
        ]

        if (
            drop_nan
            and not np.isfinite(
                X_window
            ).all()
        ):
            continue

        if target_array is not None:

            if multi_step:

                target_end = (
                    sequence_end
                    + horizon
                )

                y_window = target_array[
                    sequence_end:target_end
                ]

                target_index = (
                    target_end - 1
                )

            else:

                target_index = (
                    sequence_end
                    + horizon
                    - 1
                )

                y_window = target_array[
                    target_index
                ]

            if (
                drop_nan
                and not np.isfinite(
                    y_window
                ).all()
            ):
                continue

            y_values.append(
                np.asarray(
                    y_window,
                    dtype=dtype,
                )
            )

        else:

            target_index = (
                sequence_end
                + horizon
                - 1
            )

        X_values.append(
            X_window
        )

        target_indices.append(
            target_index
        )

    if X_values:

        X = np.stack(
            X_values
        ).astype(
            dtype
        )

    else:

        X = np.empty(
            (
                0,
                sequence_length,
                len(features),
            ),
            dtype=dtype,
        )

    if target_array is not None:

        if y_values:

            y = np.stack(
                y_values
            ).astype(
                dtype
            )

        else:

            if multi_step:

                y = np.empty(
                    (
                        0,
                        horizon,
                        len(targets),
                    ),
                    dtype=dtype,
                )

            else:

                y = np.empty(
                    (
                        0,
                        len(targets),
                    ),
                    dtype=dtype,
                )

    else:

        y = None

    return SequenceDataset(
        X=X,
        y=y,

        target_indices=np.asarray(
            target_indices,
            dtype=np.int64,
        ),

        feature_columns=features,
        target_columns=targets,

        sequence_length=sequence_length,
        horizon=horizon,
    )


# =====================================================================
# Supervised lag dataset
# =====================================================================


def create_lagged_supervised_dataset(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    lags: Sequence[int] = (
        1,
        2,
        3,
        6,
        12,
        24,
    ),
    horizon: int = 1,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Create classical ML forecasting data.

    Example:

        temperature_lag_1
        temperature_lag_2
        ...
        target_t_plus_1

    Useful for:
        XGBoost
        LightGBM
        Random Forest
        linear models
    """

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column not found: "
            f"{target_column}"
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    df = dataframe.copy()

    for lag in lags:

        if lag <= 0:
            raise ValueError(
                "Lag values must be positive."
            )

        df[
            f"{target_column}_lag_{lag}"
        ] = df[
            target_column
        ].shift(
            int(lag)
        )

    df[
        f"{target_column}_target_t_plus_{horizon}"
    ] = df[
        target_column
    ].shift(
        -horizon
    )

    if drop_na:

        required = [
            f"{target_column}_lag_{lag}"
            for lag in lags
        ]

        required.append(
            f"{target_column}_target_t_plus_{horizon}"
        )

        df = df.dropna(
            subset=required
        )

    return df.reset_index(
        drop=True
    )


# =====================================================================
# Rolling / expanding backtesting
# =====================================================================


def rolling_backtest_splits(
    n_samples: int,
    *,
    initial_train_size: int,
    validation_size: int,
    step_size: Optional[int] = None,
    expanding: bool = True,
    max_train_size: Optional[int] = None,
) -> Iterator[BacktestSplit]:

    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    if initial_train_size <= 0:
        raise ValueError(
            "initial_train_size must be positive."
        )

    if validation_size <= 0:
        raise ValueError(
            "validation_size must be positive."
        )

    if step_size is None:
        step_size = validation_size

    if step_size <= 0:
        raise ValueError(
            "step_size must be positive."
        )

    fold = 0

    train_end = (
        initial_train_size
    )

    while (
        train_end
        + validation_size
        <= n_samples
    ):

        if expanding:

            train_start = 0

            if max_train_size is not None:

                train_start = max(
                    0,
                    train_end
                    - max_train_size,
                )

        else:

            window = (
                max_train_size
                if max_train_size is not None
                else initial_train_size
            )

            train_start = max(
                0,
                train_end - window,
            )

        validation_start = (
            train_end
        )

        validation_end = (
            validation_start
            + validation_size
        )

        yield BacktestSplit(
            fold=fold,

            train_start=train_start,
            train_end=train_end,

            validation_start=(
                validation_start
            ),

            validation_end=(
                validation_end
            ),

            train_indices=np.arange(
                train_start,
                train_end,
                dtype=np.int64,
            ),

            validation_indices=np.arange(
                validation_start,
                validation_end,
                dtype=np.int64,
            ),
        )

        fold += 1

        train_end += (
            step_size
        )


# =====================================================================
# Forecast metrics
# =====================================================================


def calculate_forecast_metrics(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    *,
    epsilon: float = 1e-8,
) -> ForecastMetrics:

    true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(
        -1
    )

    pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(
        -1
    )

    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have "
            "the same number of values."
        )

    valid = (
        np.isfinite(true)
        & np.isfinite(pred)
    )

    true = true[
        valid
    ]

    pred = pred[
        valid
    ]

    if true.size == 0:
        raise ValueError(
            "No finite observations available "
            "for metric calculation."
        )

    error = (
        pred - true
    )

    absolute_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    mae = float(
        np.mean(
            absolute_error
        )
    )

    mse = float(
        np.mean(
            squared_error
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    # MAPE excludes near-zero targets.
    nonzero = (
        np.abs(true)
        > epsilon
    )

    if nonzero.any():

        mape = float(
            np.mean(
                absolute_error[nonzero]
                / np.abs(
                    true[nonzero]
                )
            )
            * 100.0
        )

    else:

        mape = float(
            "nan"
        )

    denominator = (
        np.abs(true)
        + np.abs(pred)
    )

    valid_smape = (
        denominator > epsilon
    )

    if valid_smape.any():

        smape = float(
            np.mean(
                (
                    2.0
                    * absolute_error[
                        valid_smape
                    ]
                )
                / denominator[
                    valid_smape
                ]
            )
            * 100.0
        )

    else:

        smape = 0.0

    absolute_true_sum = float(
        np.sum(
            np.abs(true)
        )
    )

    if absolute_true_sum > epsilon:

        wape = float(
            np.sum(
                absolute_error
            )
            / absolute_true_sum
            * 100.0
        )

    else:

        wape = float(
            "nan"
        )

    bias = float(
        np.mean(
            error
        )
    )

    total_variance = float(
        np.sum(
            (
                true
                - np.mean(true)
            )
            ** 2
        )
    )

    if total_variance > epsilon:

        r2 = float(
            1.0
            - (
                np.sum(
                    squared_error
                )
                / total_variance
            )
        )

    else:

        r2 = float(
            "nan"
        )

    return ForecastMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        mape=mape,
        smape=smape,
        wape=wape,
        bias=bias,
        r2=r2,
    )


# =====================================================================
# Residuals
# =====================================================================


def calculate_residuals(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
) -> np.ndarray:

    true = np.asarray(
        y_true,
        dtype=float,
    )

    pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    return (
        true - pred
    )


# =====================================================================
# Prediction intervals
# =====================================================================


def residual_prediction_interval(
    predictions: np.ndarray | Sequence[float],
    calibration_residuals: np.ndarray | Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Simple empirical residual prediction interval.

    Residuals should come from validation/calibration data,
    not training residuals when possible.
    """

    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    prediction_array = np.asarray(
        predictions,
        dtype=float,
    )

    residuals = np.asarray(
        calibration_residuals,
        dtype=float,
    ).reshape(
        -1
    )

    residuals = residuals[
        np.isfinite(
            residuals
        )
    ]

    if residuals.size == 0:
        raise ValueError(
            "No finite calibration residuals."
        )

    alpha = (
        1.0 - confidence
    )

    lower_residual = np.quantile(
        residuals,
        alpha / 2.0,
    )

    upper_residual = np.quantile(
        residuals,
        1.0 - alpha / 2.0,
    )

    lower = (
        prediction_array
        + lower_residual
    )

    upper = (
        prediction_array
        + upper_residual
    )

    return (
        lower,
        upper,
    )


# =====================================================================
# Naive forecasting baselines
# =====================================================================


def naive_last_value_forecast(
    history: Sequence[float] | np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    values = np.asarray(
        history,
        dtype=float,
    ).reshape(
        -1
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size == 0:
        raise ValueError(
            "History contains no finite values."
        )

    return np.full(
        horizon,
        finite[-1],
        dtype=float,
    )


def seasonal_naive_forecast(
    history: Sequence[float] | np.ndarray,
    *,
    horizon: int,
    season_length: int,
) -> np.ndarray:

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    if season_length <= 0:
        raise ValueError(
            "season_length must be positive."
        )

    values = np.asarray(
        history,
        dtype=float,
    ).reshape(
        -1
    )

    if len(values) < season_length:
        raise ValueError(
            "History must contain at least one "
            "complete seasonal cycle."
        )

    season = values[
        -season_length:
    ]

    repeats = int(
        np.ceil(
            horizon
            / season_length
        )
    )

    return np.tile(
        season,
        repeats,
    )[:horizon]


# =====================================================================
# Forecast alignment
# =====================================================================


def align_predictions(
    dataframe: pd.DataFrame,
    *,
    target_indices: Sequence[int],
    predictions: Sequence[float] | np.ndarray,
    prediction_column: str = "prediction",
) -> pd.DataFrame:

    indices = np.asarray(
        target_indices,
        dtype=int,
    )

    pred = np.asarray(
        predictions,
    )

    if pred.ndim > 1:

        if (
            pred.ndim == 2
            and pred.shape[1] == 1
        ):

            pred = pred[:, 0]

        else:

            raise ValueError(
                "align_predictions expects one prediction "
                "per target index."
            )

    if len(indices) != len(pred):
        raise ValueError(
            "target_indices and predictions "
            "must have equal length."
        )

    if (
        len(indices) > 0
        and (
            indices.min() < 0
            or indices.max() >= len(dataframe)
        )
    ):
        raise IndexError(
            "target_indices contain values outside "
            "the dataframe."
        )

    result = dataframe.copy()

    result[
        prediction_column
    ] = np.nan

    column_index = (
        result.columns.get_loc(
            prediction_column
        )
    )

    result.iloc[
        indices,
        column_index,
    ] = pred

    return result


# =====================================================================
# Difference / inverse difference
# =====================================================================


def difference_series(
    values: Sequence[float] | np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:

    if order <= 0:
        raise ValueError(
            "order must be positive."
        )

    result = np.asarray(
        values,
        dtype=float,
    )

    for _ in range(
        order
    ):
        result = np.diff(
            result
        )

    return result


def inverse_first_difference(
    last_observation: float,
    differences: Sequence[float] | np.ndarray,
) -> np.ndarray:

    diffs = np.asarray(
        differences,
        dtype=float,
    )

    return (
        float(last_observation)
        + np.cumsum(
            diffs
        )
    )


# =====================================================================
# Trend utilities
# =====================================================================


def linear_trend(
    values: Sequence[float] | np.ndarray,
) -> float:

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(
        -1
    )

    valid = np.isfinite(
        array
    )

    if valid.sum() < 2:
        return 0.0

    y = array[
        valid
    ]

    x = np.arange(
        len(array),
        dtype=float,
    )[valid]

    x_mean = float(
        np.mean(x)
    )

    y_mean = float(
        np.mean(y)
    )

    denominator = float(
        np.sum(
            (
                x - x_mean
            )
            ** 2
        )
    )

    if denominator <= 0:
        return 0.0

    numerator = float(
        np.sum(
            (
                x - x_mean
            )
            * (
                y - y_mean
            )
        )
    )

    return (
        numerator
        / denominator
    )


# =====================================================================
# Outlier clipping
# =====================================================================


def clip_outliers_iqr(
    dataframe: pd.DataFrame,
    *,
    columns: Sequence[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:

    if multiplier <= 0:
        raise ValueError(
            "multiplier must be positive."
        )

    df = dataframe.copy()

    for column in columns:

        if column not in df.columns:
            raise ValueError(
                f"Column not found: {column}"
            )

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        q1 = values.quantile(
            0.25
        )

        q3 = values.quantile(
            0.75
        )

        iqr = (
            q3 - q1
        )

        if (
            not np.isfinite(iqr)
            or iqr <= 0
        ):
            continue

        lower = (
            q1
            - multiplier * iqr
        )

        upper = (
            q3
            + multiplier * iqr
        )

        df[column] = values.clip(
            lower=lower,
            upper=upper,
        )

    return df


# =====================================================================
# Finite-value sanitizer
# =====================================================================


def sanitize_numeric_features(
    dataframe: pd.DataFrame,
    *,
    fill_value: Optional[float] = None,
) -> pd.DataFrame:

    df = dataframe.copy()

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if fill_value is not None:

        df[numeric_columns] = (
            df[numeric_columns]
            .fillna(
                float(
                    fill_value
                )
            )
        )

    return df


# =====================================================================
# Reproducibility
# =====================================================================


def set_random_seed(
    seed: int = 42,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    # Optional PyTorch support.
    try:
        import torch

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                seed
            )

    except ImportError:
        pass


# =====================================================================
# Forecast horizon helpers
# =====================================================================


def hours_to_steps(
    hours: float,
    *,
    sampling_interval_hours: float,
) -> int:

    if hours <= 0:
        raise ValueError(
            "hours must be positive."
        )

    if sampling_interval_hours <= 0:
        raise ValueError(
            "sampling_interval_hours must be positive."
        )

    return max(
        1,
        int(
            np.ceil(
                hours
                / sampling_interval_hours
            )
        ),
    )


def steps_to_hours(
    steps: int,
    *,
    sampling_interval_hours: float,
) -> float:

    if steps < 0:
        raise ValueError(
            "steps cannot be negative."
        )

    if sampling_interval_hours <= 0:
        raise ValueError(
            "sampling_interval_hours must be positive."
        )

    return float(
        steps
        * sampling_interval_hours
    )


# =====================================================================
# Convenience preparation pipeline
# =====================================================================


def prepare_forecasting_dataset(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    timestamp_column: str = "timestamp",
    sequence_length: int = 24,
    horizon: int = 1,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Convenience preparation function.

    Important:
    Splitting happens before scaler fitting.

    This prevents validation/test information from leaking into
    training normalization statistics.
    """

    df = prepare_time_series(
        dataframe,
        timestamp_column=timestamp_column,
    )

    split = chronological_split(
        df,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        timestamp_column=timestamp_column,
    )

    feature_scaler = StandardScaler()

    feature_scaler.fit(
        split.train[
            list(feature_columns)
        ]
    )

    def scale_frame(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:

        result = frame.copy()

        result[
            list(feature_columns)
        ] = feature_scaler.transform(
            result[
                list(feature_columns)
            ]
        )

        return result

    train_scaled = scale_frame(
        split.train
    )

    validation_scaled = scale_frame(
        split.validation
    )

    test_scaled = scale_frame(
        split.test
    )

    train_sequences = (
        create_forecasting_sequences(
            train_scaled,
            feature_columns=feature_columns,
            target_columns=target_columns,
            sequence_length=sequence_length,
            horizon=horizon,
        )
    )

    validation_sequences = (
        create_forecasting_sequences(
            validation_scaled,
            feature_columns=feature_columns,
            target_columns=target_columns,
            sequence_length=sequence_length,
            horizon=horizon,
        )
    )

    test_sequences = (
        create_forecasting_sequences(
            test_scaled,
            feature_columns=feature_columns,
            target_columns=target_columns,
            sequence_length=sequence_length,
            horizon=horizon,
        )
    )

    return {
        "split": split,

        "feature_scaler":
            feature_scaler,

        "train_sequences":
            train_sequences,

        "validation_sequences":
            validation_sequences,

        "test_sequences":
            test_sequences,
    }


# =====================================================================
# Demo
# =====================================================================


if __name__ == "__main__":

    set_random_seed(
        42
    )

    rng = np.random.default_rng(
        42
    )

    rows = 500

    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="h",
        tz="UTC",
    )

    temperature = (
        60.0
        + np.linspace(
            0,
            15,
            rows,
        )
        + rng.normal(
            0,
            1.5,
            rows,
        )
    )

    vibration = (
        3.0
        + np.linspace(
            0,
            2,
            rows,
        )
        + rng.normal(
            0,
            0.2,
            rows,
        )
    )

    demo = pd.DataFrame(
        {
            "timestamp":
                timestamps,

            "temperature":
                temperature,

            "vibration_rms":
                vibration,
        }
    )

    split = chronological_split(
        demo
    )

    print(
        "Train:",
        len(split.train),
    )

    print(
        "Validation:",
        len(split.validation),
    )

    print(
        "Test:",
        len(split.test),
    )

    scaler = StandardScaler()

    scaler.fit(
        split.train[
            [
                "temperature",
                "vibration_rms",
            ]
        ]
    )

    scaled = demo.copy()

    scaled[
        [
            "temperature",
            "vibration_rms",
        ]
    ] = scaler.transform(
        demo[
            [
                "temperature",
                "vibration_rms",
            ]
        ]
    )

    sequences = (
        create_forecasting_sequences(
            scaled,

            feature_columns=[
                "temperature",
                "vibration_rms",
            ],

            target_columns=[
                "temperature",
            ],

            sequence_length=24,
            horizon=6,
            multi_step=True,
        )
    )

    print(
        "X:",
        sequences.X.shape,
    )

    print(
        "y:",
        (
            sequences.y.shape
            if sequences.y is not None
            else None
        ),
    )

    # -------------------------------------------------------------
    # Baseline evaluation example
    # -------------------------------------------------------------

    actual = np.asarray(
        [
            70,
            72,
            73,
            75,
            74,
        ],
        dtype=float,
    )

    predicted = np.asarray(
        [
            69,
            72.5,
            74,
            74,
            75,
        ],
        dtype=float,
    )

    metrics = calculate_forecast_metrics(
        actual,
        predicted,
    )

    print(
        metrics.to_dict()
    )

    # -------------------------------------------------------------
    # Backtesting example
    # -------------------------------------------------------------

    for fold in rolling_backtest_splits(
        n_samples=500,
        initial_train_size=300,
        validation_size=50,
        step_size=50,
    ):

        print(
            "Fold:",
            fold.fold,
            "train:",
            (
                fold.train_start,
                fold.train_end,
            ),
            "validation:",
            (
                fold.validation_start,
                fold.validation_end,
            ),
        )