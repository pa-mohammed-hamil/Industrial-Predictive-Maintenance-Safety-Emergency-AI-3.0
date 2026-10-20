"""
Industrial Asset Degradation Model
==================================

Estimates machine/component degradation from multivariate telemetry.

Capabilities
------------
- Sensor normalization
- Weighted health/degradation scoring
- Exponential smoothing
- Degradation trend estimation
- Degradation rate
- Health index
- Time-to-critical estimation
- Confidence scoring
- Sensor contribution analysis
- Batch/history analysis

Designed for:
    Predictive Maintenance
        -> degradation_model.py
        -> rul_prediction.py
        -> maintenance_forecasting.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class DegradationState(str, Enum):
    HEALTHY = "healthy"
    EARLY_DEGRADATION = "early_degradation"
    MODERATE_DEGRADATION = "moderate_degradation"
    SEVERE_DEGRADATION = "severe_degradation"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SensorConfig:
    """
    Configuration for one degradation-related sensor.

    healthy_min / healthy_max:
        Expected healthy operating range.

    critical_min / critical_max:
        Values considered critically degraded.

    weight:
        Contribution to total degradation score.
    """

    name: str

    healthy_min: Optional[float] = None
    healthy_max: Optional[float] = None

    critical_min: Optional[float] = None
    critical_max: Optional[float] = None

    weight: float = 1.0


@dataclass(frozen=True)
class SensorReading:
    name: str
    value: float
    timestamp: Optional[datetime] = None
    quality: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class DegradationResult:
    timestamp: str

    degradation_score: float
    health_index: float

    degradation_state: DegradationState

    degradation_rate: float
    trend: str

    estimated_time_to_critical: Optional[float]

    confidence: float

    sensor_scores: Dict[str, float]
    sensor_contributions: Dict[str, float]

    dominant_degradation_signals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "degradation_score": round(
                self.degradation_score, 4
            ),
            "health_index": round(
                self.health_index, 4
            ),
            "degradation_state": self.degradation_state.value,
            "degradation_rate": round(
                self.degradation_rate, 6
            ),
            "trend": self.trend,
            "estimated_time_to_critical": (
                round(self.estimated_time_to_critical, 2)
                if self.estimated_time_to_critical is not None
                else None
            ),
            "confidence": round(self.confidence, 4),
            "sensor_scores": {
                key: round(value, 4)
                for key, value in self.sensor_scores.items()
            },
            "sensor_contributions": {
                key: round(value, 4)
                for key, value
                in self.sensor_contributions.items()
            },
            "dominant_degradation_signals":
                self.dominant_degradation_signals,
        }


class DegradationModel:
    """
    Explainable industrial degradation estimator.

    Score interpretation
    --------------------
    0.0 -> healthy
    1.0 -> critical degradation
    """

    def __init__(
        self,
        sensor_configs: Sequence[SensorConfig],
        smoothing_alpha: float = 0.30,
        critical_score: float = 0.90,
        trend_window: int = 10,
    ) -> None:

        if not sensor_configs:
            raise ValueError(
                "At least one SensorConfig is required."
            )

        if not 0 < smoothing_alpha <= 1:
            raise ValueError(
                "smoothing_alpha must be within (0, 1]."
            )

        if not 0 < critical_score <= 1:
            raise ValueError(
                "critical_score must be within (0, 1]."
            )

        if trend_window < 2:
            raise ValueError(
                "trend_window must be >= 2."
            )

        self.configs = {
            config.name: config
            for config in sensor_configs
        }

        self.smoothing_alpha = smoothing_alpha
        self.critical_score = critical_score
        self.trend_window = trend_window

        self._history: List[float] = []
        self._smoothed_score: Optional[float] = None

    # ==============================================================
    # Public API
    # ==============================================================

    def predict(
        self,
        readings: Sequence[SensorReading],
    ) -> DegradationResult:

        if not readings:
            raise ValueError(
                "Sensor readings cannot be empty."
            )

        sensor_scores: Dict[str, float] = {}
        qualities: Dict[str, float] = {}

        for reading in readings:

            config = self.configs.get(reading.name)

            if config is None:
                continue

            if not isfinite(reading.value):
                continue

            score = self._sensor_degradation(
                reading.value,
                config,
            )

            sensor_scores[reading.name] = score

            qualities[reading.name] = self._clamp(
                reading.quality
            )

        if not sensor_scores:
            raise ValueError(
                "No valid configured sensor readings supplied."
            )

        raw_score, contributions = (
            self._aggregate_degradation(
                sensor_scores,
                qualities,
            )
        )

        degradation_score = self._smooth(raw_score)

        self._history.append(degradation_score)

        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        degradation_rate = self._calculate_rate()

        trend = self._determine_trend(
            degradation_rate
        )

        state = self._classify_state(
            degradation_score
        )

        health_index = self._clamp(
            1.0 - degradation_score
        )

        time_to_critical = (
            self._estimate_time_to_critical(
                degradation_score,
                degradation_rate,
            )
        )

        confidence = self._calculate_confidence(
            sensor_scores,
            qualities,
        )

        dominant = [
            name
            for name, _ in sorted(
                contributions.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]

        return DegradationResult(
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            degradation_score=degradation_score,
            health_index=health_index,
            degradation_state=state,
            degradation_rate=degradation_rate,
            trend=trend,
            estimated_time_to_critical=time_to_critical,
            confidence=confidence,
            sensor_scores=sensor_scores,
            sensor_contributions=contributions,
            dominant_degradation_signals=dominant,
        )

    def predict_dict(
        self,
        readings: Mapping[str, float],
    ) -> Dict[str, Any]:

        sensor_readings = [
            SensorReading(
                name=name,
                value=float(value),
            )
            for name, value in readings.items()
        ]

        return self.predict(
            sensor_readings
        ).to_dict()

    def analyze_history(
        self,
        history: Sequence[Mapping[str, float]],
    ) -> List[Dict[str, Any]]:
        """
        Process sequential telemetry observations.

        Each history item represents one timestamp/sample.
        """

        results: List[Dict[str, Any]] = []

        for sample in history:
            results.append(
                self.predict_dict(sample)
            )

        return results

    def reset(self) -> None:
        """Reset temporal degradation state."""

        self._history.clear()
        self._smoothed_score = None

    # ==============================================================
    # Sensor degradation
    # ==============================================================

    def _sensor_degradation(
        self,
        value: float,
        config: SensorConfig,
    ) -> float:

        upper_score = self._upper_degradation(
            value,
            config,
        )

        lower_score = self._lower_degradation(
            value,
            config,
        )

        return max(
            upper_score,
            lower_score,
        )

    def _upper_degradation(
        self,
        value: float,
        config: SensorConfig,
    ) -> float:

        healthy = config.healthy_max
        critical = config.critical_max

        if healthy is None:
            return 0.0

        if value <= healthy:
            return 0.0

        if critical is None:
            # Soft nonlinear degradation when no critical
            # engineering threshold is configured.
            excess = abs(value - healthy)

            denominator = max(
                abs(healthy),
                1.0,
            )

            return self._clamp(
                excess / denominator
            )

        if critical <= healthy:
            raise ValueError(
                f"critical_max must exceed healthy_max "
                f"for sensor '{config.name}'."
            )

        if value >= critical:
            return 1.0

        return self._clamp(
            (value - healthy)
            / (critical - healthy)
        )

    def _lower_degradation(
        self,
        value: float,
        config: SensorConfig,
    ) -> float:

        healthy = config.healthy_min
        critical = config.critical_min

        if healthy is None:
            return 0.0

        if value >= healthy:
            return 0.0

        if critical is None:

            deficit = abs(
                healthy - value
            )

            denominator = max(
                abs(healthy),
                1.0,
            )

            return self._clamp(
                deficit / denominator
            )

        if critical >= healthy:
            raise ValueError(
                f"critical_min must be below healthy_min "
                f"for sensor '{config.name}'."
            )

        if value <= critical:
            return 1.0

        return self._clamp(
            (healthy - value)
            / (healthy - critical)
        )

    # ==============================================================
    # Aggregation
    # ==============================================================

    def _aggregate_degradation(
        self,
        scores: Mapping[str, float],
        qualities: Mapping[str, float],
    ) -> tuple[float, Dict[str, float]]:

        weighted_total = 0.0
        total_weight = 0.0

        raw_contributions: Dict[str, float] = {}

        for name, score in scores.items():

            config = self.configs[name]

            quality = qualities.get(
                name,
                1.0,
            )

            weight = max(
                config.weight,
                0.0,
            ) * quality

            contribution = (
                score * weight
            )

            weighted_total += contribution
            total_weight += weight

            raw_contributions[name] = contribution

        if total_weight <= 0:
            return 0.0, {
                name: 0.0
                for name in scores
            }

        result = self._clamp(
            weighted_total / total_weight
        )

        contributions = {
            name: value / total_weight
            for name, value
            in raw_contributions.items()
        }

        return result, contributions

    # ==============================================================
    # Temporal degradation
    # ==============================================================

    def _smooth(
        self,
        current: float,
    ) -> float:

        if self._smoothed_score is None:
            self._smoothed_score = current
            return current

        self._smoothed_score = (
            self.smoothing_alpha * current
            + (
                1.0 - self.smoothing_alpha
            )
            * self._smoothed_score
        )

        return self._clamp(
            self._smoothed_score
        )

    def _calculate_rate(self) -> float:

        if len(self._history) < 2:
            return 0.0

        values = np.asarray(
            self._history[
                -self.trend_window:
            ],
            dtype=float,
        )

        if len(values) < 2:
            return 0.0

        x = np.arange(
            len(values),
            dtype=float,
        )

        # Linear regression slope.
        slope = np.polyfit(
            x,
            values,
            1,
        )[0]

        if not np.isfinite(slope):
            return 0.0

        return float(slope)

    @staticmethod
    def _determine_trend(
        rate: float,
    ) -> str:

        if rate >= 0.02:
            return "rapidly_degrading"

        if rate >= 0.005:
            return "degrading"

        if rate <= -0.01:
            return "improving"

        return "stable"

    # ==============================================================
    # Prognostics
    # ==============================================================

    def _estimate_time_to_critical(
        self,
        current_score: float,
        rate: float,
    ) -> Optional[float]:
        """
        Estimate remaining samples until critical degradation.

        This value becomes hours/cycles only when each prediction
        invocation represents one hour/cycle.
        """

        if current_score >= self.critical_score:
            return 0.0

        if rate <= 0:
            return None

        remaining = (
            self.critical_score
            - current_score
        )

        estimate = remaining / rate

        if not isfinite(estimate):
            return None

        return max(
            0.0,
            float(estimate),
        )

    # ==============================================================
    # Classification
    # ==============================================================

    def _classify_state(
        self,
        score: float,
    ) -> DegradationState:

        if score >= self.critical_score:
            return DegradationState.CRITICAL

        if score >= 0.75:
            return DegradationState.SEVERE_DEGRADATION

        if score >= 0.50:
            return DegradationState.MODERATE_DEGRADATION

        if score >= 0.25:
            return DegradationState.EARLY_DEGRADATION

        return DegradationState.HEALTHY

    # ==============================================================
    # Confidence
    # ==============================================================

    def _calculate_confidence(
        self,
        scores: Mapping[str, float],
        qualities: Mapping[str, float],
    ) -> float:

        configured_count = len(
            self.configs
        )

        available_count = len(
            scores
        )

        coverage = (
            available_count / configured_count
            if configured_count
            else 0.0
        )

        average_quality = (
            sum(qualities.values())
            / len(qualities)
            if qualities
            else 0.0
        )

        history_factor = min(
            len(self._history)
            / max(
                self.trend_window,
                1,
            ),
            1.0,
        )

        confidence = (
            0.50 * coverage
            + 0.35 * average_quality
            + 0.15 * history_factor
        )

        return self._clamp(
            confidence
        )

    # ==============================================================
    # Utility
    # ==============================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )


# ======================================================================
# Default industrial model
# ======================================================================

def create_default_degradation_model() -> DegradationModel:
    """
    Example configuration.

    Thresholds MUST be replaced with equipment/OEM-specific
    engineering limits for real deployments.
    """

    configs = [

        SensorConfig(
            name="temperature",
            healthy_max=70.0,
            critical_max=110.0,
            weight=1.2,
        ),

        SensorConfig(
            name="vibration_rms",
            healthy_max=4.5,
            critical_max=12.0,
            weight=1.4,
        ),

        SensorConfig(
            name="motor_current",
            healthy_max=30.0,
            critical_max=55.0,
            weight=1.0,
        ),

        SensorConfig(
            name="pressure",
            healthy_min=80.0,
            healthy_max=120.0,
            critical_min=50.0,
            critical_max=160.0,
            weight=1.0,
        ),

        SensorConfig(
            name="energy_consumption",
            healthy_max=100.0,
            critical_max=160.0,
            weight=0.8,
        ),
    ]

    return DegradationModel(
        sensor_configs=configs,
        smoothing_alpha=0.30,
        critical_score=0.90,
        trend_window=10,
    )


# ======================================================================
# Example
# ======================================================================

if __name__ == "__main__":

    model = create_default_degradation_model()

    telemetry_history = [

        {
            "temperature": 68,
            "vibration_rms": 3.8,
            "motor_current": 27,
            "pressure": 101,
            "energy_consumption": 92,
        },

        {
            "temperature": 74,
            "vibration_rms": 5.2,
            "motor_current": 31,
            "pressure": 98,
            "energy_consumption": 105,
        },

        {
            "temperature": 82,
            "vibration_rms": 6.8,
            "motor_current": 35,
            "pressure": 94,
            "energy_consumption": 116,
        },

        {
            "temperature": 91,
            "vibration_rms": 8.5,
            "motor_current": 40,
            "pressure": 89,
            "energy_consumption": 130,
        },

        {
            "temperature": 99,
            "vibration_rms": 10.1,
            "motor_current": 46,
            "pressure": 82,
            "energy_consumption": 145,
        },
    ]

    results = model.analyze_history(
        telemetry_history
    )

    from pprint import pprint

    pprint(results[-1])