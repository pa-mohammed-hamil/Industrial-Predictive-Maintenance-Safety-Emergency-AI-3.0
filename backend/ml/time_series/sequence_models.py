"""
Industrial Sequence Models
==========================

Deep-learning sequence models for industrial telemetry forecasting,
predictive maintenance, degradation modelling and RUL estimation.

Models:
- LSTM
- GRU
- Bidirectional LSTM
- CNN-LSTM
- Residual LSTM
- Attention LSTM
- Multi-task sequence model

Supports:
- single-step regression
- multi-step forecasting
- multi-target forecasting
- sequence-to-vector prediction
- sequence-to-sequence prediction
- failure probability
- degradation score
- RUL prediction
- dropout
- normalization
- residual connections
- attention
- gradient clipping
- early stopping
- checkpointing

Input:
    [batch, sequence_length, features]

Typical output:
    [batch, horizon, targets]

Requires:
    torch
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "sequence_models.py requires PyTorch. "
        "Install it with `pip install torch`."
    ) from exc


# =====================================================================
# Enums
# =====================================================================


class SequenceModelType(str, Enum):
    LSTM = "lstm"
    GRU = "gru"
    BIDIRECTIONAL_LSTM = "bidirectional_lstm"
    CNN_LSTM = "cnn_lstm"
    RESIDUAL_LSTM = "residual_lstm"
    ATTENTION_LSTM = "attention_lstm"
    MULTITASK_LSTM = "multitask_lstm"


class SequenceTask(str, Enum):
    FORECASTING = "forecasting"
    RUL = "rul"
    DEGRADATION = "degradation"
    FAILURE = "failure"
    MULTITASK = "multitask"


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class SequenceModelConfig:
    input_size: int

    hidden_size: int = 128
    num_layers: int = 2

    horizon: int = 1
    output_size: int = 1

    dropout: float = 0.20
    bidirectional: bool = False

    use_layer_norm: bool = True

    cnn_channels: int = 64
    cnn_kernel_size: int = 3

    attention_size: int = 128

    task: SequenceTask = SequenceTask.FORECASTING

    def validate(self) -> None:
        if self.input_size <= 0:
            raise ValueError(
                "input_size must be positive."
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if self.horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        if self.output_size <= 0:
            raise ValueError(
                "output_size must be positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    patience: int = 12
    min_delta: float = 1e-5

    gradient_clip: float = 1.0

    num_workers: int = 0

    device: Optional[str] = None

    checkpoint_path: Optional[str] = None


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(
        default_factory=list
    )

    validation_loss: list[float] = field(
        default_factory=list
    )

    best_epoch: int = -1
    best_validation_loss: float = math.inf
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "validation_loss":
                self.validation_loss,
            "best_epoch": self.best_epoch,
            "best_validation_loss":
                self.best_validation_loss,
            "stopped_early":
                self.stopped_early,
        }


# =====================================================================
# Dataset
# =====================================================================


class SequenceDataset(Dataset):
    """
    Dataset for pre-built sequence windows.

    X:
        [samples, sequence_length, features]

    y:
        [samples, output]
        or
        [samples, horizon, output]
    """

    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> None:

        self.X = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.as_tensor(
            y,
            dtype=torch.float32,
        )

        if self.X.ndim != 3:
            raise ValueError(
                "X must have shape "
                "[samples, sequence_length, features]."
            )

        if len(self.X) != len(self.y):
            raise ValueError(
                "X and y must contain the same "
                "number of samples."
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return (
            self.X[index],
            self.y[index],
        )


# =====================================================================
# Output helper
# =====================================================================


class ForecastHead(nn.Module):

    def __init__(
        self,
        input_size: int,
        horizon: int,
        output_size: int,
        dropout: float,
    ) -> None:

        super().__init__()

        hidden = max(
            input_size // 2,
            output_size,
        )

        self.horizon = horizon
        self.output_size = output_size

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden,
                horizon * output_size,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output = self.network(
            x
        )

        return output.view(
            x.size(0),
            self.horizon,
            self.output_size,
        )


# =====================================================================
# LSTM
# =====================================================================


class LSTMForecaster(nn.Module):

    def __init__(
        self,
        config: SequenceModelConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        lstm_dropout = (
            config.dropout
            if config.num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=config.bidirectional,
        )

        recurrent_size = (
            config.hidden_size
            * (
                2
                if config.bidirectional
                else 1
            )
        )

        self.norm = (
            nn.LayerNorm(
                recurrent_size
            )
            if config.use_layer_norm
            else nn.Identity()
        )

        self.head = ForecastHead(
            input_size=recurrent_size,
            horizon=config.horizon,
            output_size=config.output_size,
            dropout=config.dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence, _ = self.lstm(
            x
        )

        representation = sequence[
            :,
            -1,
            :,
        ]

        return self.norm(
            representation
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        representation = self.encode(
            x
        )

        return self.head(
            representation
        )


# =====================================================================
# GRU
# =====================================================================


class GRUForecaster(nn.Module):

    def __init__(
        self,
        config: SequenceModelConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        gru_dropout = (
            config.dropout
            if config.num_layers > 1
            else 0.0
        )

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=config.bidirectional,
        )

        recurrent_size = (
            config.hidden_size
            * (
                2
                if config.bidirectional
                else 1
            )
        )

        self.norm = (
            nn.LayerNorm(
                recurrent_size
            )
            if config.use_layer_norm
            else nn.Identity()
        )

        self.head = ForecastHead(
            recurrent_size,
            config.horizon,
            config.output_size,
            config.dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence, _ = self.gru(
            x
        )

        representation = sequence[
            :,
            -1,
            :,
        ]

        return self.norm(
            representation
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.head(
            self.encode(x)
        )


# =====================================================================
# Bidirectional LSTM
# =====================================================================


class BidirectionalLSTMForecaster(
    LSTMForecaster
):

    def __init__(
        self,
        config: SequenceModelConfig,
    ) -> None:

        copied = copy.deepcopy(
            config
        )

        copied.bidirectional = True

        super().__init__(
            copied
        )


# =====================================================================
# CNN + LSTM
# =====================================================================


class CNNLSTMForecaster(nn.Module):

    def __init__(
        self,
        config: SequenceModelConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        kernel_size = max(
            1,
            config.cnn_kernel_size,
        )

        padding = (
            kernel_size // 2
        )

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=config.input_size,
                out_channels=config.cnn_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(
                config.cnn_channels
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
        )

        self.lstm = nn.LSTM(
            input_size=config.cnn_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.num_layers > 1
                else 0.0
            ),
        )

        self.norm = nn.LayerNorm(
            config.hidden_size
        )

        self.head = ForecastHead(
            config.hidden_size,
            config.horizon,
            config.output_size,
            config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # [B, T, F] -> [B, F, T]
        x = x.transpose(
            1,
            2,
        )

        x = self.conv(
            x
        )

        # [B, C, T] -> [B, T, C]
        x = x.transpose(
            1,
            2,
        )

        sequence, _ = self.lstm(
            x
        )

        representation = self.norm(
            sequence[:, -1, :]
        )

        return self.head(
            representation
        )


# =====================================================================
# Residual LSTM
# =====================================================================


class ResidualLSTMForecaster(nn.Module):

    def __init__(
        self,
        config: SequenceModelConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.input_projection = nn.Linear(
            config.input_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=config.hidden_size,
                    hidden_size=config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                for _ in range(
                    config.num_layers
                )
            ]
        )

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    config.hidden_size
                )
                for _ in range(
                    config.num_layers
                )
            ]
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.head = ForecastHead(
            config.hidden_size,
            config.horizon,
            config.output_size,
            config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.input_projection(
            x
        )

        for recurrent, norm in zip(
            self.layers,
            self.norms,
        ):

            residual = x

            x, _ = recurrent(
                x
            )

            x = norm(
                x + residual
            )

            x = self.dropout(
                x
            )

        return self.head(
            x[:, -1, :]
        )


# =====================================================================
# Attention
# =====================================================================


class TemporalAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        attention_size: int,
    ) -> None:

        super().__init__()

        self.projection = nn.Linear(
            hidden_size,
            attention_size,
        )

        self.score = nn.Linear(
            attention_size,
            1,
            bias=False,
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        energy = torch.tanh(
            self.projection(
                sequence
            )
        )

        scores = self.score(
            energy
        ).squeeze(
            -1
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        context = torch.sum(
            sequence
            * weights.unsqueeze(-1),
            dim=1,
        )

        return (
            context,
            weights,
        )


class AttentionLSTMForecaster(nn.Module):

    def __init__(
        self,
        config: SequenceModelConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.num_layers > 1
                else 0.0
            ),
            bidirectional=config.bidirectional,
        )

        recurrent_size = (
            config.hidden_size
            * (
                2
                if config.bidirectional
                else 1
            )
        )

        self.attention = TemporalAttention(
            recurrent_size,
            config.attention_size,
        )

        self.norm = nn.LayerNorm(
            recurrent_size
        )

        self.head = ForecastHead(
            recurrent_size,
            config.horizon,
            config.output_size,
            config.dropout,
        )

        self.last_attention_weights: Optional[
            torch.Tensor
        ] = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence, _ = self.lstm(
            x
        )

        context, weights = self.attention(
            sequence
        )

        self.last_attention_weights = (
            weights.detach()
        )

        context = self.norm(
            context
        )

        return self.head(
            context
        )


# =====================================================================
# Multi-task predictive-maintenance LSTM
# =====================================================================


class MultiTaskLSTM(nn.Module):
    """
    Shared telemetry encoder with dedicated heads for:

    - sensor forecasting
    - degradation score
    - failure probability
    - RUL

    This is useful when the same telemetry sequence provides evidence
    for several predictive-maintenance tasks.
    """

    def __init__(
        self,
        config: SequenceModelConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.encoder = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.num_layers > 1
                else 0.0
            ),
        )

        self.norm = nn.LayerNorm(
            config.hidden_size
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.forecast_head = ForecastHead(
            config.hidden_size,
            config.horizon,
            config.output_size,
            config.dropout,
        )

        self.degradation_head = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                config.hidden_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_size // 2,
                1,
            ),
            nn.Sigmoid(),
        )

        self.failure_head = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                config.hidden_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_size // 2,
                1,
            ),
        )

        self.rul_head = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                config.hidden_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_size // 2,
                1,
            ),
            nn.Softplus(),
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence, _ = self.encoder(
            x
        )

        representation = sequence[
            :,
            -1,
            :,
        ]

        representation = self.norm(
            representation
        )

        return self.dropout(
            representation
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        representation = self.encode(
            x
        )

        return {
            "forecast":
                self.forecast_head(
                    representation
                ),

            "degradation":
                self.degradation_head(
                    representation
                ).squeeze(-1),

            # Raw logits. Apply sigmoid for inference.
            "failure_logits":
                self.failure_head(
                    representation
                ).squeeze(-1),

            "rul":
                self.rul_head(
                    representation
                ).squeeze(-1),
        }


# =====================================================================
# Multi-task loss
# =====================================================================


@dataclass
class MultiTaskLossWeights:
    forecast: float = 1.0
    degradation: float = 0.8
    failure: float = 1.0
    rul: float = 1.0


class MultiTaskPredictiveMaintenanceLoss(
    nn.Module
):

    def __init__(
        self,
        weights: Optional[
            MultiTaskLossWeights
        ] = None,
    ) -> None:

        super().__init__()

        self.weights = (
            weights
            or MultiTaskLossWeights()
        )

        self.regression_loss = (
            nn.SmoothL1Loss()
        )

        self.failure_loss = (
            nn.BCEWithLogitsLoss()
        )

    def forward(
        self,
        predictions: Mapping[
            str,
            torch.Tensor,
        ],
        targets: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        dict[str, float],
    ]:

        device = next(
            iter(
                predictions.values()
            )
        ).device

        total = torch.zeros(
            (),
            device=device,
        )

        losses: dict[
            str,
            torch.Tensor,
        ] = {}

        if (
            "forecast" in predictions
            and "forecast" in targets
        ):

            loss = self.regression_loss(
                predictions["forecast"],
                targets["forecast"],
            )

            losses["forecast"] = loss

            total = (
                total
                + self.weights.forecast
                * loss
            )

        if (
            "degradation" in predictions
            and "degradation" in targets
        ):

            loss = self.regression_loss(
                predictions["degradation"],
                targets["degradation"],
            )

            losses["degradation"] = loss

            total = (
                total
                + self.weights.degradation
                * loss
            )

        if (
            "failure_logits" in predictions
            and "failure" in targets
        ):

            loss = self.failure_loss(
                predictions[
                    "failure_logits"
                ],
                targets["failure"],
            )

            losses["failure"] = loss

            total = (
                total
                + self.weights.failure
                * loss
            )

        if (
            "rul" in predictions
            and "rul" in targets
        ):

            loss = self.regression_loss(
                predictions["rul"],
                targets["rul"],
            )

            losses["rul"] = loss

            total = (
                total
                + self.weights.rul
                * loss
            )

        if not losses:
            raise ValueError(
                "No matching prediction/target "
                "pairs were supplied."
            )

        return (
            total,
            {
                name: float(
                    value.detach().cpu()
                )
                for name, value
                in losses.items()
            },
        )


# =====================================================================
# Factory
# =====================================================================


def create_sequence_model(
    model_type: SequenceModelType | str,
    config: SequenceModelConfig,
) -> nn.Module:

    if not isinstance(
        model_type,
        SequenceModelType,
    ):
        model_type = SequenceModelType(
            str(model_type).lower()
        )

    if model_type == SequenceModelType.LSTM:
        return LSTMForecaster(
            config
        )

    if model_type == SequenceModelType.GRU:
        return GRUForecaster(
            config
        )

    if (
        model_type
        == SequenceModelType.BIDIRECTIONAL_LSTM
    ):
        return BidirectionalLSTMForecaster(
            config
        )

    if (
        model_type
        == SequenceModelType.CNN_LSTM
    ):
        return CNNLSTMForecaster(
            config
        )

    if (
        model_type
        == SequenceModelType.RESIDUAL_LSTM
    ):
        return ResidualLSTMForecaster(
            config
        )

    if (
        model_type
        == SequenceModelType.ATTENTION_LSTM
    ):
        return AttentionLSTMForecaster(
            config
        )

    if (
        model_type
        == SequenceModelType.MULTITASK_LSTM
    ):
        return MultiTaskLSTM(
            config
        )

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


# =====================================================================
# Standard trainer
# =====================================================================


class SequenceModelTrainer:

    def __init__(
        self,
        model: nn.Module,
        *,
        config: Optional[
            TrainingConfig
        ] = None,
        criterion: Optional[
            nn.Module
        ] = None,
    ) -> None:

        self.model = model

        self.config = (
            config
            or TrainingConfig()
        )

        self.device = self._resolve_device(
            self.config.device
        )

        self.model.to(
            self.device
        )

        self.criterion = (
            criterion
            or nn.SmoothL1Loss()
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def fit(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[
            Dataset
        ] = None,
    ) -> TrainingHistory:

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        validation_loader = (
            DataLoader(
                validation_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
            if validation_dataset is not None
            else None
        )

        history = TrainingHistory()

        best_state = copy.deepcopy(
            self.model.state_dict()
        )

        best_loss = math.inf
        patience_counter = 0

        for epoch in range(
            self.config.epochs
        ):

            train_loss = self._train_epoch(
                train_loader
            )

            history.train_loss.append(
                train_loss
            )

            if validation_loader is not None:
                validation_loss = (
                    self._evaluate_loader(
                        validation_loader
                    )
                )
            else:
                validation_loss = train_loss

            history.validation_loss.append(
                validation_loss
            )

            improved = (
                validation_loss
                < best_loss
                - self.config.min_delta
            )

            if improved:
                best_loss = validation_loss
                history.best_epoch = epoch
                history.best_validation_loss = (
                    validation_loss
                )

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                patience_counter = 0

                if self.config.checkpoint_path:
                    self.save_checkpoint(
                        self.config.checkpoint_path,
                        epoch=epoch,
                        validation_loss=(
                            validation_loss
                        ),
                    )

            else:
                patience_counter += 1

            if (
                validation_loader is not None
                and patience_counter
                >= self.config.patience
            ):
                history.stopped_early = True
                break

        self.model.load_state_dict(
            best_state
        )

        return history

    def _train_epoch(
        self,
        loader: DataLoader,
    ) -> float:

        self.model.train()

        total_loss = 0.0
        samples = 0

        for X, y in loader:

            X = X.to(
                self.device
            )

            y = y.to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            prediction = self.model(
                X
            )

            prediction, y = (
                self._align_prediction_target(
                    prediction,
                    y,
                )
            )

            loss = self.criterion(
                prediction,
                y,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite training loss detected."
                )

            loss.backward()

            if (
                self.config.gradient_clip
                and self.config.gradient_clip > 0
            ):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            batch_size = X.size(
                0
            )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * batch_size
            )

            samples += batch_size

        return (
            total_loss
            / max(samples, 1)
        )

    @torch.no_grad()
    def _evaluate_loader(
        self,
        loader: DataLoader,
    ) -> float:

        self.model.eval()

        total_loss = 0.0
        samples = 0

        for X, y in loader:

            X = X.to(
                self.device
            )

            y = y.to(
                self.device
            )

            prediction = self.model(
                X
            )

            prediction, y = (
                self._align_prediction_target(
                    prediction,
                    y,
                )
            )

            loss = self.criterion(
                prediction,
                y,
            )

            batch_size = X.size(
                0
            )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * batch_size
            )

            samples += batch_size

        return (
            total_loss
            / max(samples, 1)
        )

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:

        self.model.eval()

        tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        if tensor.ndim != 3:
            raise ValueError(
                "X must have shape [N, T, F]."
            )

        loader = DataLoader(
            tensor,
            batch_size=(
                batch_size
                or self.config.batch_size
            ),
            shuffle=False,
        )

        predictions = []

        for batch in loader:

            batch = batch.to(
                self.device
            )

            prediction = self.model(
                batch
            )

            if isinstance(
                prediction,
                Mapping,
            ):
                raise TypeError(
                    "Standard predict() does not "
                    "support multi-task dictionary output."
                )

            predictions.append(
                prediction.cpu().numpy()
            )

        if not predictions:
            return np.empty(
                (0,),
                dtype=np.float32,
            )

        return np.concatenate(
            predictions,
            axis=0,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: Optional[int] = None,
        validation_loss: Optional[
            float
        ] = None,
    ) -> None:

        destination = Path(
            path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "epoch":
                epoch,

            "validation_loss":
                validation_loss,
        }

        torch.save(
            payload,
            destination,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:

        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            load_optimizer
            and "optimizer_state_dict"
            in checkpoint
        ):
            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        return checkpoint

    @staticmethod
    def _align_prediction_target(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Allows common target layouts:

        prediction [B,H,O]
        target     [B,H,O]

        prediction [B,1,O]
        target     [B,O]

        prediction [B,1,1]
        target     [B]
        """

        if prediction.shape == target.shape:
            return prediction, target

        if (
            prediction.ndim == 3
            and prediction.shape[1] == 1
            and target.ndim == 2
            and prediction.shape[0]
            == target.shape[0]
            and prediction.shape[2]
            == target.shape[1]
        ):
            return (
                prediction[:, 0, :],
                target,
            )

        if (
            prediction.ndim == 3
            and prediction.shape[1:] == (1, 1)
            and target.ndim == 1
        ):
            return (
                prediction[:, 0, 0],
                target,
            )

        raise ValueError(
            "Prediction and target shapes are "
            f"incompatible: {tuple(prediction.shape)} "
            f"vs {tuple(target.shape)}."
        )

    @staticmethod
    def _resolve_device(
        requested: Optional[str],
    ) -> torch.device:

        if requested:
            return torch.device(
                requested
            )

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device(
                "mps"
            )

        return torch.device(
            "cpu"
        )


# =====================================================================
# Multi-task inference
# =====================================================================


@torch.no_grad()
def predict_multitask(
    model: MultiTaskLSTM,
    X: np.ndarray | torch.Tensor,
    *,
    device: Optional[str] = None,
) -> dict[str, np.ndarray]:

    target_device = (
        torch.device(device)
        if device
        else next(
            model.parameters()
        ).device
    )

    model.eval()

    tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=target_device,
    )

    predictions = model(
        tensor
    )

    return {
        "forecast":
            predictions[
                "forecast"
            ].cpu().numpy(),

        "degradation":
            predictions[
                "degradation"
            ].cpu().numpy(),

        "failure_probability":
            torch.sigmoid(
                predictions[
                    "failure_logits"
                ]
            ).cpu().numpy(),

        "rul":
            predictions[
                "rul"
            ].cpu().numpy(),
    }


# =====================================================================
# Model metadata / registry support
# =====================================================================


def model_metadata(
    model: nn.Module,
    config: SequenceModelConfig,
    *,
    model_type: SequenceModelType | str,
) -> dict[str, Any]:

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "model_type": (
            model_type.value
            if isinstance(
                model_type,
                SequenceModelType,
            )
            else str(model_type)
        ),
        "framework": "pytorch",
        "config": {
            **asdict(config),
            "task": config.task.value,
        },
        "parameters": parameters,
        "trainable_parameters":
            trainable_parameters,
    }


# =====================================================================
# Default models
# =====================================================================


def create_default_lstm(
    input_size: int,
    *,
    horizon: int = 24,
    output_size: int = 1,
) -> LSTMForecaster:

    return LSTMForecaster(
        SequenceModelConfig(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            horizon=horizon,
            output_size=output_size,
            dropout=0.20,
            task=SequenceTask.FORECASTING,
        )
    )


def create_default_gru(
    input_size: int,
    *,
    horizon: int = 24,
    output_size: int = 1,
) -> GRUForecaster:

    return GRUForecaster(
        SequenceModelConfig(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            horizon=horizon,
            output_size=output_size,
            dropout=0.20,
        )
    )


def create_default_attention_lstm(
    input_size: int,
    *,
    horizon: int = 24,
    output_size: int = 1,
) -> AttentionLSTMForecaster:

    return AttentionLSTMForecaster(
        SequenceModelConfig(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            horizon=horizon,
            output_size=output_size,
            dropout=0.20,
            attention_size=128,
        )
    )


# =====================================================================
# Demo
# =====================================================================


if __name__ == "__main__":

    torch.manual_seed(
        42
    )

    np.random.seed(
        42
    )

    samples = 256
    sequence_length = 24
    features = 5
    horizon = 6

    X = np.random.randn(
        samples,
        sequence_length,
        features,
    ).astype(
        np.float32
    )

    # Example multi-step temperature target.
    y = np.random.randn(
        samples,
        horizon,
        1,
    ).astype(
        np.float32
    )

    model_config = SequenceModelConfig(
        input_size=features,
        hidden_size=64,
        num_layers=2,
        horizon=horizon,
        output_size=1,
        dropout=0.20,
    )

    model = create_sequence_model(
        SequenceModelType.ATTENTION_LSTM,
        model_config,
    )

    train_dataset = SequenceDataset(
        X[:200],
        y[:200],
    )

    validation_dataset = SequenceDataset(
        X[200:],
        y[200:],
    )

    trainer = SequenceModelTrainer(
        model,
        config=TrainingConfig(
            epochs=3,
            batch_size=32,
            learning_rate=1e-3,
            patience=2,
        ),
    )

    history = trainer.fit(
        train_dataset,
        validation_dataset,
    )

    predictions = trainer.predict(
        X[200:210]
    )

    print(
        "Prediction shape:",
        predictions.shape,
    )

    print(
        "Training history:",
        history.to_dict(),
    )

    print(
        "Metadata:",
        model_metadata(
            model,
            model_config,
            model_type=(
                SequenceModelType.ATTENTION_LSTM
            ),
        ),
    )