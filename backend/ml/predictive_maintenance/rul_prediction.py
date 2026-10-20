"""
Remaining Useful Life (RUL) Prediction
======================================

Industrial prognostics module for estimating remaining useful life.

Inputs:
- current degradation score
- degradation rate
- failure probability
- anomaly score
- asset age / operating hours
- telemetry health indicators
- optional historical degradation observations

Outputs:
- RUL in hours/days/cycles
- lower/upper uncertainty bounds
- confidence
- health state
- degradation trend
- failure horizon
- maintenance horizon
- explanation / dominant factors

RUL is an estimate for maintenance planning and should not be used as
the sole basis for safety-critical shutdown decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


# ==============================================================
# Enums
# ==============================================================


class RULRiskLevel(str, Enum):
    HEALTHY = "healthy"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DegradationTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    RAPIDLY_DEGRADING = "rapidly_degrading"


# ==============================================================
# Input / output
# ==============================================================


@dataclass(frozen=True)
class RULInput:
    asset_id: str

    degradation_score: float
    degradation_rate: float = 0.0

    failure_probability: float = 0.0
    anomaly_score: float = 0.0

    operating_hours: float = 0.0

    expected_life_hours: Optional[float] = None

    utilization: float = 1.0
    criticality: float = 0.5

    # Optional sequential degradation history.
    degradation_history: Sequence[float] = field(
        default_factory=tuple
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class RULPrediction:
    asset_id: str
    timestamp: str

    rul_hours: float
    rul_days: float

    lower_bound_hours: float
    upper_bound_hours: float

    confidence: float

    risk_level: RULRiskLevel
    trend: DegradationTrend

    degradation_score: float
    degradation_rate: float

    failure_probability: float
    anomaly_score: float

    estimated_failure_at: Optional[str]
    recommended_maintenance_within_hours: float

    dominant_factors: list[str]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "asset_id": self.asset_id,
            "timestamp": self.timestamp,

            "rul_hours": round(
                self.rul_hours,
                2,
            ),

            "rul_days": round(
                self.rul_days,
                2,
            ),

            "lower_bound_hours": round(
                self.lower_bound_hours,
                2,
            ),

            "upper_bound_hours": round(
                self.upper_bound_hours,
                2,
            ),

            "confidence": round(
                self.confidence,
                4,
            ),

            "risk_level": self.risk_level.value,
            "trend": self.trend.value,

            "degradation_score": round(
                self.degradation_score,
                4,
            ),

            "degradation_rate": round(
                self.degradation_rate,
                6,
            ),

            "failure_probability": round(
                self.failure_probability,
                4,
            ),

            "anomaly_score": round(
                self.anomaly_score,
                4,
            ),

            "estimated_failure_at":
                self.estimated_failure_at,

            "recommended_maintenance_within_hours":
                round(
                    self.recommended_maintenance_within_hours,
                    2,
                ),

            "dominant_factors":
                self.dominant_factors,
        }


# ==============================================================
# RUL Model
# ==============================================================


class RULPredictionModel:
    """
    Explainable RUL estimator.

    Combines three independent estimates:

    1. degradation-based RUL
    2. expected-lifetime RUL
    3. failure/anomaly-adjusted RUL

    A trained survival model, LSTM, Transformer, XGBoost or Weibull
    model can later replace/augment this baseline.
    """

    def __init__(
        self,
        *,
        critical_degradation: float = 0.90,
        default_expected_life_hours: float = 10000.0,
        minimum_rul_hours: float = 0.0,
        maximum_rul_hours: float = 100000.0,
        maintenance_safety_factor: float = 0.70,
    ) -> None:

        if not 0 < critical_degradation <= 1:
            raise ValueError(
                "critical_degradation must be within (0, 1]."
            )

        if default_expected_life_hours <= 0:
            raise ValueError(
                "default_expected_life_hours must be positive."
            )

        if not 0 < maintenance_safety_factor <= 1:
            raise ValueError(
                "maintenance_safety_factor must be within (0, 1]."
            )

        self.critical_degradation = (
            critical_degradation
        )

        self.default_expected_life_hours = (
            default_expected_life_hours
        )

        self.minimum_rul_hours = max(
            minimum_rul_hours,
            0.0,
        )

        self.maximum_rul_hours = max(
            maximum_rul_hours,
            self.minimum_rul_hours,
        )

        self.maintenance_safety_factor = (
            maintenance_safety_factor
        )

    # ==========================================================
    # Prediction
    # ==========================================================

    def predict(
        self,
        data: RULInput,
    ) -> RULPrediction:

        degradation = self._clamp(
            data.degradation_score
        )

        failure_probability = self._clamp(
            data.failure_probability
        )

        anomaly_score = self._clamp(
            data.anomaly_score
        )

        utilization = max(
            float(data.utilization),
            0.01,
        )

        criticality = self._clamp(
            data.criticality
        )

        degradation_rate = self._resolve_degradation_rate(
            explicit_rate=data.degradation_rate,
            history=data.degradation_history,
        )

        expected_life = (
            float(data.expected_life_hours)
            if data.expected_life_hours is not None
            else self.default_expected_life_hours
        )

        expected_life = max(
            expected_life,
            1.0,
        )

        # ------------------------------------------------------
        # Estimate 1: degradation progression
        # ------------------------------------------------------

        degradation_rul = self._degradation_based_rul(
            degradation=degradation,
            degradation_rate=degradation_rate,
            expected_life=expected_life,
        )

        # ------------------------------------------------------
        # Estimate 2: design / expected life
        # ------------------------------------------------------

        lifecycle_rul = self._lifecycle_rul(
            operating_hours=data.operating_hours,
            expected_life=expected_life,
        )

        # ------------------------------------------------------
        # Combine estimates
        # ------------------------------------------------------

        rul = self._combine_rul_estimates(
            degradation_rul=degradation_rul,
            lifecycle_rul=lifecycle_rul,
            degradation_rate=degradation_rate,
        )

        # ------------------------------------------------------
        # Adjust for risk evidence
        # ------------------------------------------------------

        rul = self._risk_adjustment(
            rul=rul,
            failure_probability=failure_probability,
            anomaly_score=anomaly_score,
        )

        # ------------------------------------------------------
        # Utilization adjustment
        # ------------------------------------------------------

        rul = self._utilization_adjustment(
            rul,
            utilization,
        )

        # ------------------------------------------------------
        # Criticality adjustment
        # ------------------------------------------------------

        rul = self._criticality_adjustment(
            rul,
            criticality,
        )

        rul = self._bound_rul(
            rul
        )

        trend = self._trend(
            degradation_rate
        )

        risk_level = self._risk_level(
            rul
        )

        confidence = self._confidence(
            data=data,
            degradation_rate=degradation_rate,
        )

        lower, upper = self._uncertainty_interval(
            rul=rul,
            confidence=confidence,
        )

        maintenance_horizon = max(
            0.0,
            rul * self.maintenance_safety_factor,
        )

        failure_at = self._estimated_failure_time(
            rul
        )

        factors = self._dominant_factors(
            degradation=degradation,
            degradation_rate=degradation_rate,
            failure_probability=failure_probability,
            anomaly_score=anomaly_score,
            utilization=utilization,
        )

        return RULPrediction(
            asset_id=data.asset_id,
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            rul_hours=rul,
            rul_days=rul / 24.0,

            lower_bound_hours=lower,
            upper_bound_hours=upper,

            confidence=confidence,

            risk_level=risk_level,
            trend=trend,

            degradation_score=degradation,
            degradation_rate=degradation_rate,

            failure_probability=failure_probability,
            anomaly_score=anomaly_score,

            estimated_failure_at=failure_at,

            recommended_maintenance_within_hours=(
                maintenance_horizon
            ),

            dominant_factors=factors,
        )

    # ==========================================================
    # Dictionary API
    # ==============================================================

    def predict_dict(
        self,
        *,
        asset_id: str,
        degradation_score: float,
        degradation_rate: float = 0.0,
        failure_probability: float = 0.0,
        anomaly_score: float = 0.0,
        operating_hours: float = 0.0,
        expected_life_hours: Optional[float] = None,
        utilization: float = 1.0,
        criticality: float = 0.5,
        degradation_history: Sequence[float] = (),
    ) -> Dict[str, Any]:

        data = RULInput(
            asset_id=asset_id,
            degradation_score=degradation_score,
            degradation_rate=degradation_rate,
            failure_probability=failure_probability,
            anomaly_score=anomaly_score,
            operating_hours=operating_hours,
            expected_life_hours=expected_life_hours,
            utilization=utilization,
            criticality=criticality,
            degradation_history=degradation_history,
        )

        return self.predict(
            data
        ).to_dict()

    # ==========================================================
    # Degradation rate
    # ==============================================================

    def _resolve_degradation_rate(
        self,
        *,
        explicit_rate: float,
        history: Sequence[float],
    ) -> float:

        if explicit_rate != 0:
            return float(
                explicit_rate
            )

        if len(history) < 2:
            return 0.0

        values = np.asarray(
            history,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
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

        if not np.isfinite(slope):
            return 0.0

        return float(
            slope
        )

    # ==========================================================
    # Degradation RUL
    # ==============================================================

    def _degradation_based_rul(
        self,
        *,
        degradation: float,
        degradation_rate: float,
        expected_life: float,
    ) -> float:

        if degradation >= self.critical_degradation:
            return 0.0

        remaining_degradation = (
            self.critical_degradation
            - degradation
        )

        if degradation_rate > 0:

            estimate = (
                remaining_degradation
                / degradation_rate
            )

            if isfinite(estimate):
                return max(
                    0.0,
                    estimate,
                )

        # No measurable degradation slope yet.
        #
        # Use remaining health proportion as fallback.

        health_fraction = (
            remaining_degradation
            / self.critical_degradation
        )

        return max(
            0.0,
            expected_life
            * health_fraction,
        )

    # ==========================================================
    # Lifecycle RUL
    # ==============================================================

    @staticmethod
    def _lifecycle_rul(
        *,
        operating_hours: float,
        expected_life: float,
    ) -> float:

        age = max(
            float(operating_hours),
            0.0,
        )

        return max(
            0.0,
            expected_life - age,
        )

    # ==========================================================
    # Combine estimates
    # ==============================================================

    @staticmethod
    def _combine_rul_estimates(
        *,
        degradation_rul: float,
        lifecycle_rul: float,
        degradation_rate: float,
    ) -> float:

        if degradation_rate > 0:

            # When degradation trend exists, prognostic evidence
            # receives greater weight.

            return (
                0.70 * degradation_rul
                + 0.30 * lifecycle_rul
            )

        return (
            0.40 * degradation_rul
            + 0.60 * lifecycle_rul
        )

    # ==========================================================
    # Risk adjustment
    # ==============================================================

    def _risk_adjustment(
        self,
        *,
        rul: float,
        failure_probability: float,
        anomaly_score: float,
    ) -> float:

        failure_factor = (
            1.0
            - 0.55 * failure_probability
        )

        anomaly_factor = (
            1.0
            - 0.30 * anomaly_score
        )

        adjusted = (
            rul
            * failure_factor
            * anomaly_factor
        )

        # Strong agreement between failure + anomaly models.
        if (
            failure_probability >= 0.80
            and anomaly_score >= 0.80
        ):
            adjusted *= 0.75

        return max(
            adjusted,
            0.0,
        )

    # ==========================================================
    # Utilization
    # ==============================================================

    @staticmethod
    def _utilization_adjustment(
        rul: float,
        utilization: float,
    ) -> float:

        # 1.0 = expected load.
        #
        # >1 = overloaded / higher utilization.
        # <1 = lighter utilization.

        utilization = max(
            utilization,
            0.25,
        )

        return (
            rul / utilization
        )

    # ==========================================================
    # Criticality
    # ==============================================================

    @staticmethod
    def _criticality_adjustment(
        rul: float,
        criticality: float,
    ) -> float:

        # This is a conservative planning adjustment rather than
        # a physical lifetime calculation.

        planning_factor = (
            1.0
            - 0.10 * criticality
        )

        return max(
            rul * planning_factor,
            0.0,
        )

    # ==========================================================
    # Risk level
    # ==============================================================

    @staticmethod
    def _risk_level(
        rul_hours: float,
    ) -> RULRiskLevel:

        if rul_hours <= 24:
            return RULRiskLevel.CRITICAL

        if rul_hours <= 168:
            return RULRiskLevel.HIGH

        if rul_hours <= 720:
            return RULRiskLevel.MEDIUM

        if rul_hours <= 1440:
            return RULRiskLevel.LOW

        return RULRiskLevel.HEALTHY

    # ==========================================================
    # Trend
    # ==============================================================

    @staticmethod
    def _trend(
        rate: float,
    ) -> DegradationTrend:

        if rate >= 0.02:
            return (
                DegradationTrend
                .RAPIDLY_DEGRADING
            )

        if rate >= 0.005:
            return (
                DegradationTrend
                .DEGRADING
            )

        if rate <= -0.005:
            return (
                DegradationTrend
                .IMPROVING
            )

        return DegradationTrend.STABLE

    # ==========================================================
    # Confidence
    # ==============================================================

    def _confidence(
        self,
        *,
        data: RULInput,
        degradation_rate: float,
    ) -> float:

        evidence = 0.0
        possible = 6.0

        if data.degradation_score > 0:
            evidence += 1.0

        if degradation_rate != 0:
            evidence += 1.0

        if data.failure_probability > 0:
            evidence += 1.0

        if data.anomaly_score > 0:
            evidence += 1.0

        if data.operating_hours > 0:
            evidence += 1.0

        if len(data.degradation_history) >= 3:
            evidence += 1.0

        coverage = (
            evidence / possible
        )

        history_bonus = min(
            len(data.degradation_history)
            / 20.0,
            1.0,
        )

        confidence = (
            0.35
            + 0.50 * coverage
            + 0.15 * history_bonus
        )

        return self._clamp(
            confidence
        )

    # ==========================================================
    # Uncertainty
    # ==============================================================

    def _uncertainty_interval(
        self,
        *,
        rul: float,
        confidence: float,
    ) -> tuple[float, float]:

        # Low confidence produces wider uncertainty.
        uncertainty_fraction = (
            0.10
            + (1.0 - confidence) * 0.40
        )

        margin = (
            rul
            * uncertainty_fraction
        )

        lower = max(
            self.minimum_rul_hours,
            rul - margin,
        )

        upper = min(
            self.maximum_rul_hours,
            rul + margin,
        )

        return (
            lower,
            upper,
        )

    # ==========================================================
    # Failure horizon
    # ==============================================================

    @staticmethod
    def _estimated_failure_time(
        rul_hours: float,
    ) -> Optional[str]:

        if not isfinite(
            rul_hours
        ):
            return None

        from datetime import timedelta

        timestamp = (
            datetime.now(
                timezone.utc
            )
            + timedelta(
                hours=max(
                    rul_hours,
                    0.0,
                )
            )
        )

        return timestamp.isoformat()

    # ==========================================================
    # Explainability
    # ==============================================================

    @staticmethod
    def _dominant_factors(
        *,
        degradation: float,
        degradation_rate: float,
        failure_probability: float,
        anomaly_score: float,
        utilization: float,
    ) -> list[str]:

        factors = [
            (
                "degradation_score",
                degradation,
            ),
            (
                "failure_probability",
                failure_probability,
            ),
            (
                "anomaly_score",
                anomaly_score,
            ),
            (
                "degradation_rate",
                min(
                    abs(degradation_rate) * 20,
                    1.0,
                ),
            ),
            (
                "high_utilization",
                max(
                    utilization - 1.0,
                    0.0,
                ),
            ),
        ]

        factors = [
            item
            for item in factors
            if item[1] > 0
        ]

        factors.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            name
            for name, _
            in factors[:5]
        ]

    # ==========================================================
    # Bounds
    # ==============================================================

    def _bound_rul(
        self,
        value: float,
    ) -> float:

        if not isfinite(
            value
        ):
            return (
                self.maximum_rul_hours
            )

        return max(
            self.minimum_rul_hours,
            min(
                self.maximum_rul_hours,
                float(value),
            ),
        )

    # ==========================================================
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


# ==============================================================
# Factory
# ==============================================================


def create_default_rul_model(
) -> RULPredictionModel:

    return RULPredictionModel(
        critical_degradation=0.90,
        default_expected_life_hours=10000.0,
        minimum_rul_hours=0.0,
        maximum_rul_hours=100000.0,
        maintenance_safety_factor=0.70,
    )


# ==============================================================
# Convenience API
# ==============================================================


def predict_rul(
    *,
    asset_id: str,
    degradation_score: float,
    degradation_rate: float = 0.0,
    failure_probability: float = 0.0,
    anomaly_score: float = 0.0,
    operating_hours: float = 0.0,
    expected_life_hours: Optional[float] = None,
    utilization: float = 1.0,
    criticality: float = 0.5,
    degradation_history: Sequence[float] = (),
) -> Dict[str, Any]:

    model = create_default_rul_model()

    return model.predict_dict(
        asset_id=asset_id,
        degradation_score=degradation_score,
        degradation_rate=degradation_rate,
        failure_probability=failure_probability,
        anomaly_score=anomaly_score,
        operating_hours=operating_hours,
        expected_life_hours=expected_life_hours,
        utilization=utilization,
        criticality=criticality,
        degradation_history=degradation_history,
    )


# ==============================================================
# Local demonstration
# ==============================================================


if __name__ == "__main__":

    model = create_default_rul_model()

    history = [
        0.25,
        0.28,
        0.31,
        0.35,
        0.39,
        0.44,
        0.49,
        0.55,
        0.61,
        0.68,
    ]

    prediction = model.predict_dict(
        asset_id="MOTOR-001",

        degradation_score=0.68,

        # Use 0 here to derive rate from history.
        degradation_rate=0.0,

        failure_probability=0.72,
        anomaly_score=0.78,

        operating_hours=7200,
        expected_life_hours=10000,

        utilization=1.10,
        criticality=0.90,

        degradation_history=history,
    )

    from pprint import pprint

    pprint(prediction)