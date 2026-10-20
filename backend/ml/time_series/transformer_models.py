"""
Industrial Transformer Time-Series Models
==========================================

Transformer architectures for industrial telemetry forecasting and
predictive-maintenance intelligence.

Models
------
- TransformerForecaster
- CausalTransformerForecaster
- PatchTransformerForecaster
- AttentionPoolingTransformer
- MultiTaskTransformer

Supported tasks
---------------
- Multi-step sensor forecasting
- Multivariate forecasting
- Degradation estimation
- Failure probability estimation
- Remaining Useful Life (RUL)
- Multi-task predictive maintenance

Input
-----
    [batch, sequence_length, input_features]

Forecast output
---------------
    [batch, horizon, output_features]

Requires
--------
    torch
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "transformer_models.py requires PyTorch. "
        "Install it with `pip install torch`."
    ) from exc


# =====================================================================
# Enums
# =====================================================================


class TransformerModelType(str, Enum):
    TRANSFORMER = "transformer"
    CAUSAL_TRANSFORMER = "causal_transformer"
    PATCH_TRANSFORMER = "patch_transformer"
    ATTENTION_POOLING = "attention_pooling"
    MULTITASK_TRANSFORMER = "multitask_transformer"


class TransformerTask(str, Enum):
    FORECASTING = "forecasting"
    DEGRADATION = "degradation"
    FAILURE = "failure"
    RUL = "rul"
    MULTITASK = "multitask"


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class TransformerConfig:
    input_size: int

    d_model: int = 128
    nhead: int = 8

    num_layers: int = 4
    dim_feedforward: int = 256

    dropout: float = 0.10

    horizon: int = 24
    output_size: int = 1

    max_sequence_length: int = 2048

    activation: str = "gelu"
    norm_first: bool = True

    use_input_norm: bool = True

    # Patch transformer
    patch_size: int = 8
    patch_stride: int = 4

    # Optional static/context features.
    context_size: int = 0

    task: TransformerTask = (
        TransformerTask.FORECASTING
    )

    def validate(self) -> None:

        if self.input_size <= 0:
            raise ValueError(
                "input_size must be positive."
            )

        if self.d_model <= 0:
            raise ValueError(
                "d_model must be positive."
            )

        if self.nhead <= 0:
            raise ValueError(
                "nhead must be positive."
            )

        if self.d_model % self.nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if self.dim_feedforward <= 0:
            raise ValueError(
                "dim_feedforward must be positive."
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

        if self.max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive."
            )

        if self.patch_size <= 0:
            raise ValueError(
                "patch_size must be positive."
            )

        if self.patch_stride <= 0:
            raise ValueError(
                "patch_stride must be positive."
            )

        if self.context_size < 0:
            raise ValueError(
                "context_size cannot be negative."
            )


@dataclass
class TransformerTrainingConfig:
    epochs: int = 100
    batch_size: int = 64

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    patience: int = 12
    min_delta: float = 1e-5

    gradient_clip: float = 1.0

    warmup_steps: int = 0

    num_workers: int = 0

    device: Optional[str] = None

    checkpoint_path: Optional[str] = None


@dataclass
class TransformerTrainingHistory:
    train_loss: list[float] = field(
        default_factory=list
    )

    validation_loss: list[float] = field(
        default_factory=list
    )

    learning_rates: list[float] = field(
        default_factory=list
    )

    best_epoch: int = -1

    best_validation_loss: float = math.inf

    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:

        return {
            "train_loss":
                self.train_loss,

            "validation_loss":
                self.validation_loss,

            "learning_rates":
                self.learning_rates,

            "best_epoch":
                self.best_epoch,

            "best_validation_loss":
                self.best_validation_loss,

            "stopped_early":
                self.stopped_early,
        }


# =====================================================================
# Dataset
# =====================================================================


class TransformerTimeSeriesDataset(
    Dataset
):

    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        context: Optional[
            np.ndarray | torch.Tensor
        ] = None,
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

        self.context = None

        if context is not None:

            self.context = torch.as_tensor(
                context,
                dtype=torch.float32,
            )

            if len(self.context) != len(
                self.X
            ):
                raise ValueError(
                    "context must contain the same "
                    "number of samples as X."
                )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self,
        index: int,
    ):

        if self.context is None:

            return (
                self.X[index],
                self.y[index],
            )

        return (
            self.X[index],
            self.y[index],
            self.context[index],
        )


# =====================================================================
# Positional encoding
# =====================================================================


class SinusoidalPositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        d_model: int,
        max_length: int = 2048,
        dropout: float = 0.10,
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        encoding = torch.zeros(
            max_length,
            d_model,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        # Supports odd d_model safely.
        odd_width = encoding[
            :,
            1::2,
        ].shape[1]

        if odd_width:

            encoding[
                :,
                1::2,
            ] = torch.cos(
                position
                * div_term[
                    :odd_width
                ]
            )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence_length = x.size(
            1
        )

        if sequence_length > self.encoding.size(
            1
        ):
            raise ValueError(
                "Sequence length exceeds positional "
                "encoding capacity."
            )

        x = (
            x
            + self.encoding[
                :,
                :sequence_length,
            ].to(
                dtype=x.dtype
            )
        )

        return self.dropout(
            x
        )


# =====================================================================
# Learned positional encoding
# =====================================================================


class LearnedPositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        d_model: int,
        max_length: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.embedding = nn.Embedding(
            max_length,
            d_model,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        length = x.size(
            1
        )

        if length > self.embedding.num_embeddings:
            raise ValueError(
                "Sequence exceeds maximum configured "
                "sequence length."
            )

        positions = torch.arange(
            length,
            device=x.device,
        )

        positional = self.embedding(
            positions
        ).unsqueeze(
            0
        )

        return self.dropout(
            x + positional
        )


# =====================================================================
# Forecast head
# =====================================================================


class TransformerForecastHead(
    nn.Module
):

    def __init__(
        self,
        d_model: int,
        horizon: int,
        output_size: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.horizon = horizon
        self.output_size = output_size

        hidden = max(
            d_model // 2,
            output_size,
        )

        self.network = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Linear(
                d_model,
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

        prediction = self.network(
            x
        )

        return prediction.view(
            x.size(0),
            self.horizon,
            self.output_size,
        )


# =====================================================================
# Input embedding
# =====================================================================


class TimeSeriesEmbedding(
    nn.Module
):

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:

        super().__init__()

        self.input_norm = (
            nn.LayerNorm(
                config.input_size
            )
            if config.use_input_norm
            else nn.Identity()
        )

        self.projection = nn.Linear(
            config.input_size,
            config.d_model,
        )

        self.position = (
            SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_length=(
                    config.max_sequence_length
                ),
                dropout=config.dropout,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.input_norm(
            x
        )

        x = self.projection(
            x
        )

        return self.position(
            x
        )


# =====================================================================
# Encoder builder
# =====================================================================


def _build_encoder(
    config: TransformerConfig,
) -> nn.TransformerEncoder:

    layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=(
            config.dim_feedforward
        ),
        dropout=config.dropout,
        activation=config.activation,
        batch_first=True,
        norm_first=config.norm_first,
    )

    return nn.TransformerEncoder(
        encoder_layer=layer,
        num_layers=config.num_layers,
        norm=nn.LayerNorm(
            config.d_model
        ),
    )


# =====================================================================
# Standard Transformer
# =====================================================================


class TransformerForecaster(
    nn.Module
):

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.embedding = TimeSeriesEmbedding(
            config
        )

        self.encoder = _build_encoder(
            config
        )

        self.context_projection = None

        if config.context_size > 0:

            self.context_projection = (
                nn.Sequential(
                    nn.Linear(
                        config.context_size,
                        config.d_model,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(
                        config.d_model
                    ),
                )
            )

        self.head = TransformerForecastHead(
            d_model=config.d_model,
            horizon=config.horizon,
            output_size=config.output_size,
            dropout=config.dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
        context: Optional[
            torch.Tensor
        ] = None,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        x = self.embedding(
            x
        )

        encoded = self.encoder(
            x,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        if padding_mask is None:

            representation = encoded[
                :,
                -1,
                :,
            ]

        else:

            valid_lengths = (
                (~padding_mask)
                .sum(dim=1)
                .clamp(min=1)
            )

            batch_indices = torch.arange(
                encoded.size(0),
                device=encoded.device,
            )

            representation = encoded[
                batch_indices,
                valid_lengths - 1,
            ]

        if self.context_projection is not None:

            if context is None:
                raise ValueError(
                    "Model was configured with "
                    "context_size > 0 but context "
                    "was not supplied."
                )

            representation = (
                representation
                + self.context_projection(
                    context
                )
            )

        return representation

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[
            torch.Tensor
        ] = None,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        representation = self.encode(
            x,
            context=context,
            padding_mask=padding_mask,
        )

        return self.head(
            representation
        )


# =====================================================================
# Causal Transformer
# =====================================================================


class CausalTransformerForecaster(
    TransformerForecaster
):

    @staticmethod
    def causal_mask(
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:

        return torch.triu(
            torch.full(
                (
                    sequence_length,
                    sequence_length,
                ),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
        )

    def encode(
        self,
        x: torch.Tensor,
        context: Optional[
            torch.Tensor
        ] = None,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        embedded = self.embedding(
            x
        )

        mask = self.causal_mask(
            embedded.size(1),
            embedded.device,
        )

        encoded = self.encoder(
            embedded,
            mask=mask,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        if padding_mask is None:

            representation = encoded[
                :,
                -1,
                :,
            ]

        else:

            valid_lengths = (
                (~padding_mask)
                .sum(dim=1)
                .clamp(min=1)
            )

            indices = torch.arange(
                encoded.size(0),
                device=encoded.device,
            )

            representation = encoded[
                indices,
                valid_lengths - 1,
            ]

        if self.context_projection is not None:

            if context is None:
                raise ValueError(
                    "Context is required."
                )

            representation = (
                representation
                + self.context_projection(
                    context
                )
            )

        return representation


# =====================================================================
# Attention pooling
# =====================================================================


class AttentionPool(
    nn.Module
):

    def __init__(
        self,
        d_model: int,
    ) -> None:

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.Tanh(),
            nn.Linear(
                d_model,
                1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        scores = self.score(
            x
        ).squeeze(
            -1
        )

        if padding_mask is not None:

            scores = scores.masked_fill(
                padding_mask,
                float("-inf"),
            )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        context = torch.sum(
            x
            * weights.unsqueeze(-1),
            dim=1,
        )

        return (
            context,
            weights,
        )


class AttentionPoolingTransformer(
    nn.Module
):

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.embedding = TimeSeriesEmbedding(
            config
        )

        self.encoder = _build_encoder(
            config
        )

        self.pool = AttentionPool(
            config.d_model
        )

        self.head = TransformerForecastHead(
            config.d_model,
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
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        x = self.embedding(
            x
        )

        encoded = self.encoder(
            x,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        context, weights = self.pool(
            encoded,
            padding_mask,
        )

        self.last_attention_weights = (
            weights.detach()
        )

        return self.head(
            context
        )


# =====================================================================
# Patch Transformer
# =====================================================================


class PatchEmbedding(
    nn.Module
):

    def __init__(
        self,
        input_size: int,
        d_model: int,
        patch_size: int,
        stride: int,
    ) -> None:

        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.stride = stride

        self.projection = nn.Linear(
            patch_size * input_size,
            d_model,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected [B,T,F]."
            )

        if x.size(1) < self.patch_size:
            raise ValueError(
                "Sequence length must be >= patch_size."
            )

        patches = x.unfold(
            dimension=1,
            size=self.patch_size,
            step=self.stride,
        )

        # PyTorch unfold on [B,T,F] produces
        # [B, number_of_patches, F, patch_size].
        patches = patches.permute(
            0,
            1,
            3,
            2,
        ).contiguous()

        patches = patches.view(
            patches.size(0),
            patches.size(1),
            -1,
        )

        return self.projection(
            patches
        )


class PatchTransformerForecaster(
    nn.Module
):

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.patch_embedding = PatchEmbedding(
            input_size=config.input_size,
            d_model=config.d_model,
            patch_size=config.patch_size,
            stride=config.patch_stride,
        )

        maximum_patches = (
            max(
                1,
                (
                    config.max_sequence_length
                    - config.patch_size
                )
                // config.patch_stride
                + 1,
            )
        )

        self.position = (
            LearnedPositionalEncoding(
                d_model=config.d_model,
                max_length=maximum_patches,
                dropout=config.dropout,
            )
        )

        self.encoder = _build_encoder(
            config
        )

        self.pool = AttentionPool(
            config.d_model
        )

        self.head = TransformerForecastHead(
            config.d_model,
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

        patches = self.patch_embedding(
            x
        )

        patches = self.position(
            patches
        )

        encoded = self.encoder(
            patches
        )

        context, weights = self.pool(
            encoded
        )

        self.last_attention_weights = (
            weights.detach()
        )

        return self.head(
            context
        )


# =====================================================================
# Multi-task Transformer
# =====================================================================


class MultiTaskTransformer(
    nn.Module
):
    """
    Shared Transformer telemetry encoder.

    Outputs:
        forecast
        degradation
        failure_logits
        rul

    Failure output is deliberately returned as logits so training can
    use BCEWithLogitsLoss safely.
    """

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.embedding = TimeSeriesEmbedding(
            config
        )

        self.encoder = _build_encoder(
            config
        )

        self.pool = AttentionPool(
            config.d_model
        )

        self.forecast_head = (
            TransformerForecastHead(
                config.d_model,
                config.horizon,
                config.output_size,
                config.dropout,
            )
        )

        hidden = max(
            config.d_model // 2,
            16,
        )

        self.degradation_head = nn.Sequential(
            nn.LayerNorm(
                config.d_model
            ),
            nn.Linear(
                config.d_model,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                hidden,
                1,
            ),
            nn.Sigmoid(),
        )

        self.failure_head = nn.Sequential(
            nn.LayerNorm(
                config.d_model
            ),
            nn.Linear(
                config.d_model,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                hidden,
                1,
            ),
        )

        self.rul_head = nn.Sequential(
            nn.LayerNorm(
                config.d_model
            ),
            nn.Linear(
                config.d_model,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                hidden,
                1,
            ),
            nn.Softplus(),
        )

        self.last_attention_weights: Optional[
            torch.Tensor
        ] = None

    def encode(
        self,
        x: torch.Tensor,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        x = self.embedding(
            x
        )

        x = self.encoder(
            x,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        representation, weights = (
            self.pool(
                x,
                padding_mask,
            )
        )

        self.last_attention_weights = (
            weights.detach()
        )

        return representation

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        representation = self.encode(
            x,
            padding_mask,
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
class TransformerMultiTaskWeights:
    forecast: float = 1.0
    degradation: float = 0.8
    failure: float = 1.0
    rul: float = 1.0


class TransformerMultiTaskLoss(
    nn.Module
):

    def __init__(
        self,
        weights: Optional[
            TransformerMultiTaskWeights
        ] = None,
    ) -> None:

        super().__init__()

        self.weights = (
            weights
            or TransformerMultiTaskWeights()
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
                targets[
                    "failure"
                ],
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
                "No compatible targets were supplied."
            )

        return (
            total,
            {
                key: float(
                    value.detach().cpu()
                )
                for key, value
                in losses.items()
            },
        )


# =====================================================================
# Factory
# =====================================================================


def create_transformer_model(
    model_type: TransformerModelType | str,
    config: TransformerConfig,
) -> nn.Module:

    if not isinstance(
        model_type,
        TransformerModelType,
    ):

        model_type = (
            TransformerModelType(
                str(model_type).lower()
            )
        )

    if (
        model_type
        == TransformerModelType.TRANSFORMER
    ):
        return TransformerForecaster(
            config
        )

    if (
        model_type
        == TransformerModelType.CAUSAL_TRANSFORMER
    ):
        return CausalTransformerForecaster(
            config
        )

    if (
        model_type
        == TransformerModelType.PATCH_TRANSFORMER
    ):
        return PatchTransformerForecaster(
            config
        )

    if (
        model_type
        == TransformerModelType.ATTENTION_POOLING
    ):
        return AttentionPoolingTransformer(
            config
        )

    if (
        model_type
        == TransformerModelType.MULTITASK_TRANSFORMER
    ):
        return MultiTaskTransformer(
            config
        )

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


# =====================================================================
# Trainer
# =====================================================================


class TransformerTrainer:

    def __init__(
        self,
        model: nn.Module,
        *,
        config: Optional[
            TransformerTrainingConfig
        ] = None,
        criterion: Optional[
            nn.Module
        ] = None,
    ) -> None:

        self.model = model

        self.config = (
            config
            or TransformerTrainingConfig()
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

        self._optimizer_steps = 0

    def fit(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[
            Dataset
        ] = None,
    ) -> TransformerTrainingHistory:

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

        history = (
            TransformerTrainingHistory()
        )

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
                    self._evaluate(
                        validation_loader
                    )
                )

            else:

                validation_loss = train_loss

            history.validation_loss.append(
                validation_loss
            )

            history.learning_rates.append(
                self.optimizer.param_groups[
                    0
                ]["lr"]
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

        for batch in loader:

            X, y, context = (
                self._unpack_batch(
                    batch
                )
            )

            X = X.to(
                self.device
            )

            y = y.to(
                self.device
            )

            if context is not None:
                context = context.to(
                    self.device
                )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            self._apply_warmup()

            prediction = self._forward(
                X,
                context,
            )

            if isinstance(
                prediction,
                Mapping,
            ):
                raise TypeError(
                    "TransformerTrainer handles "
                    "single-task tensor models. "
                    "Use TransformerMultiTaskLoss "
                    "with a dedicated multi-task loop."
                )

            prediction, y = (
                self._align_target(
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
                    "Non-finite training loss."
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

            self._optimizer_steps += 1

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
            / max(
                samples,
                1,
            )
        )

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
    ) -> float:

        self.model.eval()

        total_loss = 0.0
        samples = 0

        for batch in loader:

            X, y, context = (
                self._unpack_batch(
                    batch
                )
            )

            X = X.to(
                self.device
            )

            y = y.to(
                self.device
            )

            if context is not None:
                context = context.to(
                    self.device
                )

            prediction = self._forward(
                X,
                context,
            )

            if isinstance(
                prediction,
                Mapping,
            ):
                raise TypeError(
                    "Multi-task model detected."
                )

            prediction, y = (
                self._align_target(
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
            / max(
                samples,
                1,
            )
        )

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        context: Optional[
            np.ndarray | torch.Tensor
        ] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:

        self.model.eval()

        tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        if tensor.ndim != 3:
            raise ValueError(
                "X must have shape [N,T,F]."
            )

        context_tensor = None

        if context is not None:

            context_tensor = (
                torch.as_tensor(
                    context,
                    dtype=torch.float32,
                )
            )

            if len(context_tensor) != len(
                tensor
            ):
                raise ValueError(
                    "context and X must have "
                    "equal sample counts."
                )

        predictions = []

        batch = (
            batch_size
            or self.config.batch_size
        )

        for start in range(
            0,
            len(tensor),
            batch,
        ):

            X_batch = tensor[
                start:
                start + batch
            ].to(
                self.device
            )

            context_batch = None

            if context_tensor is not None:

                context_batch = (
                    context_tensor[
                        start:
                        start + batch
                    ].to(
                        self.device
                    )
                )

            output = self._forward(
                X_batch,
                context_batch,
            )

            if isinstance(
                output,
                Mapping,
            ):
                raise TypeError(
                    "predict() expects a "
                    "single-task model."
                )

            predictions.append(
                output.cpu().numpy()
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

    def _forward(
        self,
        X: torch.Tensor,
        context: Optional[
            torch.Tensor
        ],
    ):

        if context is None:
            return self.model(
                X
            )

        try:
            return self.model(
                X,
                context=context,
            )

        except TypeError:
            return self.model(
                X
            )

    def _apply_warmup(
        self,
    ) -> None:

        warmup = (
            self.config.warmup_steps
        )

        if warmup <= 0:
            return

        if self._optimizer_steps >= warmup:
            return

        scale = (
            self._optimizer_steps + 1
        ) / warmup

        learning_rate = (
            self.config.learning_rate
            * scale
        )

        for group in (
            self.optimizer.param_groups
        ):
            group["lr"] = learning_rate

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

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "epoch":
                    epoch,

                "validation_loss":
                    validation_loss,

                "optimizer_steps":
                    self._optimizer_steps,
            },
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

        self._optimizer_steps = int(
            checkpoint.get(
                "optimizer_steps",
                0,
            )
        )

        return checkpoint

    @staticmethod
    def _unpack_batch(
        batch,
    ):

        if len(batch) == 2:
            X, y = batch
            return X, y, None

        if len(batch) == 3:
            X, y, context = batch
            return X, y, context

        raise ValueError(
            "Unexpected batch structure."
        )

    @staticmethod
    def _align_target(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

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
            and prediction.shape[1:] == (
                1,
                1,
            )
            and target.ndim == 1
        ):

            return (
                prediction[:, 0, 0],
                target,
            )

        raise ValueError(
            "Prediction and target shapes are "
            f"incompatible: "
            f"{tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}."
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
            hasattr(
                torch.backends,
                "mps",
            )
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
def predict_multitask_transformer(
    model: MultiTaskTransformer,
    X: np.ndarray | torch.Tensor,
    *,
    device: Optional[str] = None,
) -> dict[str, np.ndarray]:

    target_device = (
        torch.device(
            device
        )
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
# Model metadata
# =====================================================================


def transformer_model_metadata(
    model: nn.Module,
    config: TransformerConfig,
    *,
    model_type: TransformerModelType | str,
) -> dict[str, Any]:

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    serialized_config = asdict(
        config
    )

    serialized_config[
        "task"
    ] = config.task.value

    return {
        "model_type": (
            model_type.value
            if isinstance(
                model_type,
                TransformerModelType,
            )
            else str(
                model_type
            )
        ),

        "framework": "pytorch",

        "architecture":
            model.__class__.__name__,

        "config":
            serialized_config,

        "parameters":
            total_parameters,

        "trainable_parameters":
            trainable_parameters,
    }


# =====================================================================
# Default models
# =====================================================================


def create_default_transformer(
    input_size: int,
    *,
    horizon: int = 24,
    output_size: int = 1,
) -> TransformerForecaster:

    return TransformerForecaster(
        TransformerConfig(
            input_size=input_size,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.10,
            horizon=horizon,
            output_size=output_size,
        )
    )


def create_default_patch_transformer(
    input_size: int,
    *,
    horizon: int = 24,
    output_size: int = 1,
) -> PatchTransformerForecaster:

    return PatchTransformerForecaster(
        TransformerConfig(
            input_size=input_size,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.10,
            horizon=horizon,
            output_size=output_size,
            patch_size=8,
            patch_stride=4,
        )
    )


def create_default_multitask_transformer(
    input_size: int,
    *,
    horizon: int = 24,
    output_size: int = 1,
) -> MultiTaskTransformer:

    return MultiTaskTransformer(
        TransformerConfig(
            input_size=input_size,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.10,
            horizon=horizon,
            output_size=output_size,
            task=(
                TransformerTask.MULTITASK
            ),
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

    samples = 128
    sequence_length = 48
    features = 8
    horizon = 12

    X = np.random.randn(
        samples,
        sequence_length,
        features,
    ).astype(
        np.float32
    )

    y = np.random.randn(
        samples,
        horizon,
        1,
    ).astype(
        np.float32
    )

    config = TransformerConfig(
        input_size=features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.10,
        horizon=horizon,
        output_size=1,
    )

    model = create_transformer_model(
        TransformerModelType.TRANSFORMER,
        config,
    )

    train_dataset = (
        TransformerTimeSeriesDataset(
            X[:100],
            y[:100],
        )
    )

    validation_dataset = (
        TransformerTimeSeriesDataset(
            X[100:],
            y[100:],
        )
    )

    trainer = TransformerTrainer(
        model,
        config=TransformerTrainingConfig(
            epochs=3,
            batch_size=16,
            learning_rate=3e-4,
            patience=2,
        ),
    )

    history = trainer.fit(
        train_dataset,
        validation_dataset,
    )

    predictions = trainer.predict(
        X[100:105]
    )

    print(
        "Prediction shape:",
        predictions.shape,
    )

    print(
        "Training:",
        history.to_dict(),
    )

    print(
        "Metadata:",
        transformer_model_metadata(
            model,
            config,
            model_type=(
                TransformerModelType.TRANSFORMER
            ),
        ),
    )

    # -------------------------------------------------------------
    # Multi-task model
    # -------------------------------------------------------------

    multitask_model = (
        create_default_multitask_transformer(
            input_size=features,
            horizon=horizon,
            output_size=1,
        )
    )

    multi_output = (
        predict_multitask_transformer(
            multitask_model,
            X[:4],
        )
    )

    print(
        "Forecast:",
        multi_output[
            "forecast"
        ].shape,
    )

    print(
        "Degradation:",
        multi_output[
            "degradation"
        ].shape,
    )

    print(
        "Failure probability:",
        multi_output[
            "failure_probability"
        ].shape,
    )

    print(
        "RUL:",
        multi_output[
            "rul"
        ].shape,
    )