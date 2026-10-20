"""
Industrial Equipment Failure Prediction Engine
===============================================

Predicts machine/component failure risk from telemetry, anomaly signals,
degradation information, and operating conditions.

Outputs:
- failure probability
- risk level
- predicted failure state
- dominant risk factors
- estimated urgency
- confidence
- maintenance recommendation

This module intentionally predicts risk only. Physical shutdown/control
decisions belong to validated safety/emergency systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import exp, isfinite
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


class FailureRisk(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureState(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADING = "degrading"
    FAILURE_LIKELY = "failure_likely"
    FAILURE_IMMINENT = "failure_imminent"


@dataclass(frozen=True)
class FeatureConfig:
    name: str

    healthy_min: Optional[float] = None
    healthy_max: Optional[float] = None

    warning_min: Optional[float] = None
    warning_max: Optional[float] = None

    critical_min: Optional[float] = None
    critical_max: Optional[float] = None

    weight: float = 1.0


@dataclass(frozen=True)
class FailureInput:
    telemetry: Mapping[str, float]

    anomaly_score: float = 0.0
    degradation_score: float = 0.0

    previous_failure_probability: Optional[float] = None

    operating_hours: Optional[float] = None
    maintenance_overdue_days: float = 0.0

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FailurePrediction:
    timestamp: str

    failure_probability: float

    risk_level: FailureRisk
    failure_state: FailureState

    telemetry_risk: float
    anomaly_risk: float
    degradation_risk: float
    maintenance_risk: float

    confidence: float

    feature_scores: Dict[str, float]
    feature_contributions: Dict[str, float]

    dominant_risk_factors: list[str]

    urgency: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "failure_probability": round(
                self.failure_probability, 4
            ),
            "risk_level": self.risk_level.value,
            "failure_state": self.failure_state.value,
            "telemetry_risk": round(self.telemetry_risk, 4),
            "anomaly_risk": round(self.anomaly_risk, 4),
            "degradation_risk": round(
                self.degradation_risk, 4
            ),
            "maintenance_risk": round(
                self.maintenance_risk, 4
            ),
            "confidence": round(self.confidence, 4),
            "feature_scores": {
                key: round(value, 4)
                for key, value in self.feature_scores.items()
            },
            "feature_contributions": {
                key: round(value, 4)
                for key, value
                in self.feature_contributions.items()
            },
            "dominant_risk_factors":
                self.dominant_risk_factors,
            "urgency": self.urgency,
            "recommendation": self.recommendation,
        }


class FailurePredictionModel:
    """
    Explainable baseline failure prediction engine.

    This implementation is useful as:
    - deterministic production fallback
    - synthetic-data engine
    - model-validation baseline
    - feature pipeline
    - API inference fallback

    A trained XGBoost/LightGBM/neural model can later be plugged into
    predict_ml_probability().
    """

    def __init__(
        self,
        feature_configs: Sequence[FeatureConfig],
        telemetry_weight: float = 0.45,
        anomaly_weight: float = 0.20,
        degradation_weight: float = 0.25,
        maintenance_weight: float = 0.10,
    ) -> None:

        if not feature_configs:
            raise ValueError(
                "At least one feature configuration is required."
            )

        self.configs = {
            config.name: config
            for config in feature_configs
        }

        weights = np.asarray(
            [
                telemetry_weight,
                anomaly_weight,
                degradation_weight,
                maintenance_weight,
            ],
            dtype=float,
        )

        if np.any(weights < 0):
            raise ValueError(
                "Failure prediction weights cannot be negative."
            )

        total = float(weights.sum())

        if total <= 0:
            raise ValueError(
                "At least one prediction weight must be positive."
            )

        weights /= total

        self.telemetry_weight = float(weights[0])
        self.anomaly_weight = float(weights[1])
        self.degradation_weight = float(weights[2])
        self.maintenance_weight = float(weights[3])

    # ==========================================================
    # Main prediction
    # ==========================================================

    def predict(
        self,
        data: FailureInput,
    ) -> FailurePrediction:

        feature_scores = self._calculate_feature_scores(
            data.telemetry
        )

        telemetry_risk, contributions = (
            self._aggregate_telemetry_risk(
                feature_scores
            )
        )

        anomaly_risk = self._clamp(
            data.anomaly_score
        )

        degradation_risk = self._clamp(
            data.degradation_score
        )

        maintenance_risk = self._maintenance_risk(
            data.maintenance_overdue_days
        )

        base_probability = (
            self.telemetry_weight * telemetry_risk
            + self.anomaly_weight * anomaly_risk
            + self.degradation_weight * degradation_risk
            + self.maintenance_weight * maintenance_risk
        )

        probability = self._apply_interactions(
            base_probability=base_probability,
            telemetry_risk=telemetry_risk,
            anomaly_risk=anomaly_risk,
            degradation_risk=degradation_risk,
        )

        probability = self._apply_temporal_context(
            probability,
            data.previous_failure_probability,
        )

        probability = self._clamp(probability)

        risk = self._risk_level(
            probability
        )

        state = self._failure_state(
            probability
        )

        confidence = self._confidence(
            feature_scores
        )

        dominant = self._dominant_factors(
            contributions=contributions,
            anomaly_risk=anomaly_risk,
            degradation_risk=degradation_risk,
            maintenance_risk=maintenance_risk,
        )

        urgency = self._urgency(risk)

        recommendation = self._recommendation(
            risk,
            dominant,
        )

        return FailurePrediction(
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            failure_probability=probability,
            risk_level=risk,
            failure_state=state,
            telemetry_risk=telemetry_risk,
            anomaly_risk=anomaly_risk,
            degradation_risk=degradation_risk,
            maintenance_risk=maintenance_risk,
            confidence=confidence,
            feature_scores=feature_scores,
            feature_contributions=contributions,
            dominant_risk_factors=dominant,
            urgency=urgency,
            recommendation=recommendation,
        )

    # ==========================================================
    # Dictionary interface
    # ==========================================================

    def predict_dict(
        self,
        telemetry: Mapping[str, float],
        *,
        anomaly_score: float = 0.0,
        degradation_score: float = 0.0,
        previous_failure_probability: Optional[float] = None,
        operating_hours: Optional[float] = None,
        maintenance_overdue_days: float = 0.0,
    ) -> Dict[str, Any]:

        data = FailureInput(
            telemetry=telemetry,
            anomaly_score=anomaly_score,
            degradation_score=degradation_score,
            previous_failure_probability=(
                previous_failure_probability
            ),
            operating_hours=operating_hours,
            maintenance_overdue_days=maintenance_overdue_days,
        )

        return self.predict(data).to_dict()

    # ==========================================================
    # Feature risk
    # ==========================================================

    def _calculate_feature_scores(
        self,
        telemetry: Mapping[str, float],
    ) -> Dict[str, float]:

        scores: Dict[str, float] = {}

        for name, value in telemetry.items():

            config = self.configs.get(name)

            if config is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if not isfinite(value):
                continue

            scores[name] = self._feature_risk(
                value,
                config,
            )

        return scores

    def _feature_risk(
        self,
        value: float,
        config: FeatureConfig,
    ) -> float:

        upper = self._upper_risk(
            value,
            config,
        )

        lower = self._lower_risk(
            value,
            config,
        )

        return max(upper, lower)

    # ==========================================================
    # Upper threshold
    # ==========================================================

    def _upper_risk(
        self,
        value: float,
        config: FeatureConfig,
    ) -> float:

        healthy = config.healthy_max
        warning = config.warning_max
        critical = config.critical_max

        if healthy is None:
            return 0.0

        if value <= healthy:
            return 0.0

        if (
            critical is not None
            and value >= critical
        ):
            return 1.0

        if warning is not None:

            if warning <= healthy:
                raise ValueError(
                    f"warning_max must exceed healthy_max "
                    f"for '{config.name}'."
                )

            if value < warning:
                return self._scale(
                    value,
                    healthy,
                    warning,
                    0.0,
                    0.50,
                )

            if critical is not None:

                if critical <= warning:
                    raise ValueError(
                        f"critical_max must exceed warning_max "
                        f"for '{config.name}'."
                    )

                return self._scale(
                    value,
                    warning,
                    critical,
                    0.50,
                    1.0,
                )

            return 0.50

        if critical is not None:

            if critical <= healthy:
                raise ValueError(
                    f"critical_max must exceed healthy_max "
                    f"for '{config.name}'."
                )

            return self._scale(
                value,
                healthy,
                critical,
                0.0,
                1.0,
            )

        return self._clamp(
            (value - healthy)
            / max(abs(healthy), 1.0)
        )

    # ==========================================================
    # Lower threshold
    # ==========================================================

    def _lower_risk(
        self,
        value: float,
        config: FeatureConfig,
    ) -> float:

        healthy = config.healthy_min
        warning = config.warning_min
        critical = config.critical_min

        if healthy is None:
            return 0.0

        if value >= healthy:
            return 0.0

        if (
            critical is not None
            and value <= critical
        ):
            return 1.0

        if warning is not None:

            if warning >= healthy:
                raise ValueError(
                    f"warning_min must be below healthy_min "
                    f"for '{config.name}'."
                )

            if value > warning:
                return self._scale(
                    healthy - value,
                    0.0,
                    healthy - warning,
                    0.0,
                    0.50,
                )

            if critical is not None:

                if critical >= warning:
                    raise ValueError(
                        f"critical_min must be below warning_min "
                        f"for '{config.name}'."
                    )

                return self._scale(
                    warning - value,
                    0.0,
                    warning - critical,
                    0.50,
                    1.0,
                )

            return 0.50

        if critical is not None:

            if critical >= healthy:
                raise ValueError(
                    f"critical_min must be below healthy_min "
                    f"for '{config.name}'."
                )

            return self._scale(
                healthy - value,
                0.0,
                healthy - critical,
                0.0,
                1.0,
            )

        return self._clamp(
            (healthy - value)
            / max(abs(healthy), 1.0)
        )

    # ==========================================================
    # Telemetry aggregation
    # ==========================================================

    def _aggregate_telemetry_risk(
        self,
        scores: Mapping[str, float],
    ) -> tuple[float, Dict[str, float]]:

        if not scores:
            return 0.0, {}

        weighted_total = 0.0
        total_weight = 0.0

        raw: Dict[str, float] = {}

        for name, score in scores.items():

            weight = max(
                self.configs[name].weight,
                0.0,
            )

            contribution = (
                score * weight
            )

            raw[name] = contribution

            weighted_total += contribution
            total_weight += weight

        if total_weight <= 0:
            return 0.0, {
                name: 0.0
                for name in scores
            }

        contributions = {
            name: contribution / total_weight
            for name, contribution in raw.items()
        }

        risk = self._clamp(
            weighted_total / total_weight
        )

        return risk, contributions

    # ==========================================================
    # Maintenance
    # ==========================================================

    @staticmethod
    def _maintenance_risk(
        overdue_days: float,
    ) -> float:

        overdue_days = max(
            float(overdue_days),
            0.0,
        )

        if overdue_days == 0:
            return 0.0

        # Saturating curve.
        return min(
            1.0,
            1.0 - exp(
                -overdue_days / 30.0
            ),
        )

    # ==========================================================
    # Risk interactions
    # ==========================================================

    def _apply_interactions(
        self,
        *,
        base_probability: float,
        telemetry_risk: float,
        anomaly_risk: float,
        degradation_risk: float,
    ) -> float:

        probability = base_probability

        # Multiple independent evidence sources agreeing should
        # increase confidence in impending failure.

        if (
            telemetry_risk >= 0.70
            and anomaly_risk >= 0.70
        ):
            probability += 0.08

        if (
            telemetry_risk >= 0.70
            and degradation_risk >= 0.70
        ):
            probability += 0.08

        if (
            anomaly_risk >= 0.75
            and degradation_risk >= 0.75
        ):
            probability += 0.07

        if (
            telemetry_risk >= 0.80
            and anomaly_risk >= 0.80
            and degradation_risk >= 0.80
        ):
            probability += 0.10

        return self._clamp(
            probability
        )

    # ==========================================================
    # Temporal smoothing
    # ==========================================================

    def _apply_temporal_context(
        self,
        probability: float,
        previous: Optional[float],
    ) -> float:

        if previous is None:
            return probability

        previous = self._clamp(
            previous
        )

        # Current evidence gets higher importance.
        return self._clamp(
            0.75 * probability
            + 0.25 * previous
        )

    # ==========================================================
    # Classification
    # ==========================================================

    @staticmethod
    def _risk_level(
        probability: float,
    ) -> FailureRisk:

        if probability >= 0.85:
            return FailureRisk.CRITICAL

        if probability >= 0.70:
            return FailureRisk.HIGH

        if probability >= 0.45:
            return FailureRisk.MEDIUM

        if probability >= 0.20:
            return FailureRisk.LOW

        return FailureRisk.NORMAL

    @staticmethod
    def _failure_state(
        probability: float,
    ) -> FailureState:

        if probability >= 0.90:
            return FailureState.FAILURE_IMMINENT

        if probability >= 0.75:
            return FailureState.FAILURE_LIKELY

        if probability >= 0.50:
            return FailureState.DEGRADING

        if probability >= 0.25:
            return FailureState.WATCH

        return FailureState.HEALTHY

    # ==========================================================
    # Explainability
    # ==========================================================

    def _dominant_factors(
        self,
        *,
        contributions: Mapping[str, float],
        anomaly_risk: float,
        degradation_risk: float,
        maintenance_risk: float,
    ) -> list[str]:

        factors = [
            (
                f"telemetry:{name}",
                value * self.telemetry_weight,
            )
            for name, value in contributions.items()
        ]

        factors.extend(
            [
                (
                    "anomaly_score",
                    anomaly_risk * self.anomaly_weight,
                ),
                (
                    "degradation_score",
                    degradation_risk
                    * self.degradation_weight,
                ),
                (
                    "maintenance_overdue",
                    maintenance_risk
                    * self.maintenance_weight,
                ),
            ]
        )

        factors = [
            factor
            for factor in factors
            if factor[1] > 0
        ]

        factors.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            name
            for name, _ in factors[:5]
        ]

    # ==========================================================
    # Confidence
    # ==========================================================

    def _confidence(
        self,
        scores: Mapping[str, float],
    ) -> float:

        expected = len(self.configs)

        if expected == 0:
            return 0.0

        coverage = (
            len(scores) / expected
        )

        return self._clamp(
            0.60 + 0.40 * coverage
            if scores
            else 0.0
        )

    # ==========================================================
    # Operational recommendation
    # ==========================================================

    @staticmethod
    def _urgency(
        risk: FailureRisk,
    ) -> str:

        mapping = {
            FailureRisk.NORMAL:
                "routine",

            FailureRisk.LOW:
                "monitor",

            FailureRisk.MEDIUM:
                "schedule_inspection",

            FailureRisk.HIGH:
                "priority_maintenance",

            FailureRisk.CRITICAL:
                "immediate_engineering_review",
        }

        return mapping[risk]

    @staticmethod
    def _recommendation(
        risk: FailureRisk,
        factors: Sequence[str],
    ) -> str:

        dominant = (
            ", ".join(factors[:3])
            if factors
            else "no dominant abnormal signal"
        )

        if risk == FailureRisk.CRITICAL:
            return (
                "Request immediate engineering inspection and "
                f"validate critical signals: {dominant}."
            )

        if risk == FailureRisk.HIGH:
            return (
                "Create a priority maintenance work order and "
                f"inspect: {dominant}."
            )

        if risk == FailureRisk.MEDIUM:
            return (
                "Schedule condition-based inspection and increase "
                f"monitoring frequency for: {dominant}."
            )

        if risk == FailureRisk.LOW:
            return (
                "Continue enhanced monitoring; watch "
                f"the following signals: {dominant}."
            )

        return (
            "Continue normal condition monitoring."
        )

    # ==========================================================
    # Utilities
    # ==========================================================

    @staticmethod
    def _scale(
        value: float,
        input_min: float,
        input_max: float,
        output_min: float,
        output_max: float,
    ) -> float:

        if input_max <= input_min:
            return output_max

        ratio = (
            value - input_min
        ) / (
            input_max - input_min
        )

        ratio = max(
            0.0,
            min(1.0, ratio),
        )

        return (
            output_min
            + ratio
            * (output_max - output_min)
        )

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


# ==============================================================
# Default industrial configuration
# ==============================================================

def create_default_failure_model() -> FailurePredictionModel:

    features = [

        FeatureConfig(
            name="temperature",
            healthy_max=70.0,
            warning_max=85.0,
            critical_max=110.0,
            weight=1.30,
        ),

        FeatureConfig(
            name="vibration_rms",
            healthy_max=4.5,
            warning_max=7.0,
            critical_max=12.0,
            weight=1.50,
        ),

        FeatureConfig(
            name="motor_current",
            healthy_max=30.0,
            warning_max=40.0,
            critical_max=55.0,
            weight=1.10,
        ),

        FeatureConfig(
            name="pressure",
            healthy_min=80.0,
            healthy_max=120.0,
            warning_min=65.0,
            warning_max=140.0,
            critical_min=45.0,
            critical_max=165.0,
            weight=1.00,
        ),

        FeatureConfig(
            name="energy_consumption",
            healthy_max=100.0,
            warning_max=125.0,
            critical_max=165.0,
            weight=0.80,
        ),
    ]

    return FailurePredictionModel(
        feature_configs=features,
        telemetry_weight=0.45,
        anomaly_weight=0.20,
        degradation_weight=0.25,
        maintenance_weight=0.10,
    )


# ==============================================================
# Convenience API
# ==============================================================

def predict_failure(
    telemetry: Mapping[str, float],
    *,
    anomaly_score: float = 0.0,
    degradation_score: float = 0.0,
    previous_failure_probability: Optional[float] = None,
    maintenance_overdue_days: float = 0.0,
) -> Dict[str, Any]:

    model = create_default_failure_model()

    return model.predict_dict(
        telemetry=telemetry,
        anomaly_score=anomaly_score,
        degradation_score=degradation_score,
        previous_failure_probability=(
            previous_failure_probability
        ),
        maintenance_overdue_days=(
            maintenance_overdue_days
        ),
    )


# ==============================================================
# Local demonstration
# ==============================================================

if __name__ == "__main__":

    model = create_default_failure_model()

    telemetry = {
        "temperature": 94.0,
        "vibration_rms": 9.3,
        "motor_current": 43.0,
        "pressure": 72.0,
        "energy_consumption": 139.0,
    }

    prediction = model.predict_dict(
        telemetry=telemetry,
        anomaly_score=0.82,
        degradation_score=0.76,
        previous_failure_probability=0.64,
        maintenance_overdue_days=18,
    )

    from pprint import pprint

    pprint(prediction)