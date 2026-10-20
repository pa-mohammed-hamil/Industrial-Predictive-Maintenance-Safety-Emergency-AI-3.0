"""
Industrial Failure Risk Engine
==============================

Aggregates predictive-maintenance evidence into an explainable
asset/component failure-risk assessment.

Inputs
------
- failure probability
- degradation score and rate
- Remaining Useful Life (RUL)
- anomaly score
- telemetry risk
- maintenance condition
- operating utilization
- asset criticality
- optional previous failure-risk score

Outputs
-------
- failure risk score [0, 1]
- failure risk level
- failure state
- trend
- confidence
- component contributions
- dominant failure drivers
- escalation reasons
- recommended actions

This module provides decision-support risk scoring. It must not directly
control or shut down industrial equipment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, Mapping, Optional


# =====================================================================
# Enums
# =====================================================================


class FailureRiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureRiskState(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADING = "degrading"
    FAILURE_LIKELY = "failure_likely"
    FAILURE_IMMINENT = "failure_imminent"


class FailureRiskTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    INCREASING = "increasing"
    RAPIDLY_INCREASING = "rapidly_increasing"


# =====================================================================
# Input
# =====================================================================


@dataclass(frozen=True)
class FailureRiskInput:
    asset_id: str

    # Direct failure model output.
    failure_probability: float = 0.0

    # Degradation model outputs.
    degradation_score: float = 0.0
    degradation_rate: float = 0.0

    # RUL model output.
    rul_hours: Optional[float] = None

    # Anomaly detection.
    anomaly_score: float = 0.0

    # Aggregated telemetry condition.
    telemetry_risk: float = 0.0

    # Maintenance context.
    hours_since_maintenance: float = 0.0
    expected_maintenance_interval_hours: Optional[float] = None

    # Operating context.
    utilization: float = 1.0
    criticality: float = 0.5

    # Previous final failure-risk score.
    previous_risk_score: Optional[float] = None

    # Confidence values from upstream models.
    failure_confidence: float = 1.0
    degradation_confidence: float = 1.0
    rul_confidence: float = 1.0
    anomaly_confidence: float = 1.0
    telemetry_confidence: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# =====================================================================
# Output
# =====================================================================


@dataclass
class FailureRiskResult:
    asset_id: str
    timestamp: str

    risk_score: float
    risk_level: FailureRiskLevel
    failure_state: FailureRiskState
    trend: FailureRiskTrend

    confidence: float

    failure_probability_risk: float
    degradation_risk: float
    rul_risk: float
    anomaly_risk: float
    telemetry_risk: float
    maintenance_risk: float

    contributions: Dict[str, float]

    dominant_factors: list[str]
    escalation_reasons: list[str]
    recommended_actions: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "timestamp": self.timestamp,

            "risk_score": round(
                self.risk_score,
                4,
            ),

            "risk_level":
                self.risk_level.value,

            "failure_state":
                self.failure_state.value,

            "trend":
                self.trend.value,

            "confidence": round(
                self.confidence,
                4,
            ),

            "failure_probability_risk": round(
                self.failure_probability_risk,
                4,
            ),

            "degradation_risk": round(
                self.degradation_risk,
                4,
            ),

            "rul_risk": round(
                self.rul_risk,
                4,
            ),

            "anomaly_risk": round(
                self.anomaly_risk,
                4,
            ),

            "telemetry_risk": round(
                self.telemetry_risk,
                4,
            ),

            "maintenance_risk": round(
                self.maintenance_risk,
                4,
            ),

            "contributions": {
                key: round(value, 4)
                for key, value
                in self.contributions.items()
            },

            "dominant_factors":
                self.dominant_factors,

            "escalation_reasons":
                self.escalation_reasons,

            "recommended_actions":
                self.recommended_actions,
        }


# =====================================================================
# Failure Risk Model
# =====================================================================


class FailureRiskModel:
    """
    Explainable failure-risk aggregation model.

    Expected upstream pipeline:

        failure_prediction.py
        degradation_model.py
        rul_prediction.py
        anomaly_detection/
        telemetry

                   ↓

             failure_risk.py
    """

    def __init__(
        self,
        *,
        failure_probability_weight: float = 0.32,
        degradation_weight: float = 0.22,
        rul_weight: float = 0.18,
        anomaly_weight: float = 0.12,
        telemetry_weight: float = 0.10,
        maintenance_weight: float = 0.06,
        default_maintenance_interval_hours: float = 2000.0,
    ) -> None:

        weights = {
            "failure_probability":
                failure_probability_weight,

            "degradation":
                degradation_weight,

            "rul":
                rul_weight,

            "anomaly":
                anomaly_weight,

            "telemetry":
                telemetry_weight,

            "maintenance":
                maintenance_weight,
        }

        if any(
            value < 0
            for value in weights.values()
        ):
            raise ValueError(
                "Failure-risk weights cannot be negative."
            )

        total = sum(
            weights.values()
        )

        if total <= 0:
            raise ValueError(
                "At least one failure-risk weight must be positive."
            )

        self.weights = {
            key: value / total
            for key, value
            in weights.items()
        }

        if default_maintenance_interval_hours <= 0:
            raise ValueError(
                "default_maintenance_interval_hours "
                "must be positive."
            )

        self.default_maintenance_interval_hours = float(
            default_maintenance_interval_hours
        )

    # =================================================================
    # Main API
    # =================================================================

    def predict(
        self,
        data: FailureRiskInput,
    ) -> FailureRiskResult:

        failure_probability = self._clamp(
            data.failure_probability
        )

        degradation_risk = self._clamp(
            data.degradation_score
        )

        anomaly_risk = self._clamp(
            data.anomaly_score
        )

        telemetry_risk = self._clamp(
            data.telemetry_risk
        )

        rul_risk = self._calculate_rul_risk(
            data.rul_hours
        )

        maintenance_risk = (
            self._calculate_maintenance_risk(
                hours_since_maintenance=(
                    data.hours_since_maintenance
                ),
                expected_interval=(
                    data.expected_maintenance_interval_hours
                ),
            )
        )

        risks = {
            "failure_probability":
                failure_probability,

            "degradation":
                degradation_risk,

            "rul":
                rul_risk,

            "anomaly":
                anomaly_risk,

            "telemetry":
                telemetry_risk,

            "maintenance":
                maintenance_risk,
        }

        confidences = {
            "failure_probability":
                self._clamp(
                    data.failure_confidence
                ),

            "degradation":
                self._clamp(
                    data.degradation_confidence
                ),

            "rul":
                self._clamp(
                    data.rul_confidence
                ),

            "anomaly":
                self._clamp(
                    data.anomaly_confidence
                ),

            "telemetry":
                self._clamp(
                    data.telemetry_confidence
                ),

            # Maintenance history is deterministic evidence.
            "maintenance": 1.0,
        }

        # -------------------------------------------------------------
        # Confidence-aware weighted aggregation
        # -------------------------------------------------------------

        base_risk, contributions = (
            self._aggregate(
                risks=risks,
                confidences=confidences,
            )
        )

        escalation_reasons: list[str] = []

        # -------------------------------------------------------------
        # Cross-model correlation
        # -------------------------------------------------------------

        risk_score = (
            self._cross_signal_adjustment(
                risk=base_risk,
                risks=risks,
                degradation_rate=data.degradation_rate,
                reasons=escalation_reasons,
            )
        )

        # -------------------------------------------------------------
        # Critical failure probability
        # -------------------------------------------------------------

        risk_score = (
            self._failure_probability_override(
                risk=risk_score,
                failure_probability=failure_probability,
                reasons=escalation_reasons,
            )
        )

        # -------------------------------------------------------------
        # Short-RUL protection
        # -------------------------------------------------------------

        risk_score = self._rul_override(
            risk=risk_score,
            rul_hours=data.rul_hours,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Rapid degradation
        # -------------------------------------------------------------

        risk_score = (
            self._degradation_rate_adjustment(
                risk=risk_score,
                degradation_rate=data.degradation_rate,
                reasons=escalation_reasons,
            )
        )

        # -------------------------------------------------------------
        # Utilization
        # -------------------------------------------------------------

        risk_score = (
            self._utilization_adjustment(
                risk=risk_score,
                utilization=data.utilization,
            )
        )

        # -------------------------------------------------------------
        # Criticality
        # -------------------------------------------------------------

        risk_score = (
            self._criticality_adjustment(
                risk=risk_score,
                criticality=data.criticality,
            )
        )

        # -------------------------------------------------------------
        # Temporal smoothing
        # -------------------------------------------------------------

        risk_score = self._temporal_smoothing(
            current=risk_score,
            previous=data.previous_risk_score,
        )

        risk_score = self._clamp(
            risk_score
        )

        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        level = self._risk_level(
            risk_score
        )

        state = self._failure_state(
            risk_score
        )

        trend = self._risk_trend(
            current=risk_score,
            previous=data.previous_risk_score,
        )

        confidence = self._overall_confidence(
            confidences=confidences,
            data=data,
        )

        dominant = self._dominant_factors(
            risks=risks,
            contributions=contributions,
            degradation_rate=data.degradation_rate,
        )

        actions = self._recommended_actions(
            level=level,
            state=state,
            trend=trend,
            dominant_factors=dominant,
        )

        return FailureRiskResult(
            asset_id=data.asset_id,

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            risk_score=risk_score,
            risk_level=level,
            failure_state=state,
            trend=trend,

            confidence=confidence,

            failure_probability_risk=(
                failure_probability
            ),

            degradation_risk=(
                degradation_risk
            ),

            rul_risk=rul_risk,
            anomaly_risk=anomaly_risk,
            telemetry_risk=telemetry_risk,
            maintenance_risk=maintenance_risk,

            contributions=contributions,

            dominant_factors=dominant,

            escalation_reasons=(
                escalation_reasons
            ),

            recommended_actions=actions,
        )

    # =================================================================
    # Dictionary API
    # =================================================================

    def predict_dict(
        self,
        *,
        asset_id: str,
        failure_probability: float = 0.0,
        degradation_score: float = 0.0,
        degradation_rate: float = 0.0,
        rul_hours: Optional[float] = None,
        anomaly_score: float = 0.0,
        telemetry_risk: float = 0.0,
        hours_since_maintenance: float = 0.0,
        expected_maintenance_interval_hours: Optional[
            float
        ] = None,
        utilization: float = 1.0,
        criticality: float = 0.5,
        previous_risk_score: Optional[float] = None,
        failure_confidence: float = 1.0,
        degradation_confidence: float = 1.0,
        rul_confidence: float = 1.0,
        anomaly_confidence: float = 1.0,
        telemetry_confidence: float = 1.0,
    ) -> Dict[str, Any]:

        data = FailureRiskInput(
            asset_id=asset_id,

            failure_probability=(
                failure_probability
            ),

            degradation_score=(
                degradation_score
            ),

            degradation_rate=(
                degradation_rate
            ),

            rul_hours=rul_hours,

            anomaly_score=anomaly_score,
            telemetry_risk=telemetry_risk,

            hours_since_maintenance=(
                hours_since_maintenance
            ),

            expected_maintenance_interval_hours=(
                expected_maintenance_interval_hours
            ),

            utilization=utilization,
            criticality=criticality,

            previous_risk_score=(
                previous_risk_score
            ),

            failure_confidence=(
                failure_confidence
            ),

            degradation_confidence=(
                degradation_confidence
            ),

            rul_confidence=(
                rul_confidence
            ),

            anomaly_confidence=(
                anomaly_confidence
            ),

            telemetry_confidence=(
                telemetry_confidence
            ),
        )

        return self.predict(
            data
        ).to_dict()

    # =================================================================
    # Weighted aggregation
    # =================================================================

    def _aggregate(
        self,
        *,
        risks: Mapping[str, float],
        confidences: Mapping[str, float],
    ) -> tuple[float, Dict[str, float]]:

        weighted_total = 0.0
        effective_weight_total = 0.0

        raw_contributions: Dict[
            str,
            float
        ] = {}

        for name, risk in risks.items():

            weight = self.weights[
                name
            ]

            confidence = confidences[
                name
            ]

            effective_weight = (
                weight * confidence
            )

            contribution = (
                risk * effective_weight
            )

            raw_contributions[
                name
            ] = contribution

            weighted_total += (
                contribution
            )

            effective_weight_total += (
                effective_weight
            )

        if effective_weight_total <= 0:

            return (
                0.0,
                {
                    name: 0.0
                    for name in risks
                },
            )

        score = (
            weighted_total
            / effective_weight_total
        )

        contributions = {
            name: value
            / effective_weight_total

            for name, value
            in raw_contributions.items()
        }

        return (
            self._clamp(score),
            contributions,
        )

    # =================================================================
    # RUL risk
    # =================================================================

    @staticmethod
    def _calculate_rul_risk(
        rul_hours: Optional[float],
    ) -> float:

        if rul_hours is None:
            return 0.0

        rul = max(
            float(rul_hours),
            0.0,
        )

        if rul <= 24:
            return 1.0

        if rul <= 72:
            return 0.92

        if rul <= 168:
            return 0.80

        if rul <= 336:
            return 0.62

        if rul <= 720:
            return 0.42

        if rul <= 1440:
            return 0.22

        return 0.05

    # =================================================================
    # Maintenance risk
    # =================================================================

    def _calculate_maintenance_risk(
        self,
        *,
        hours_since_maintenance: float,
        expected_interval: Optional[float],
    ) -> float:

        interval = (
            float(expected_interval)
            if expected_interval is not None
            else self.default_maintenance_interval_hours
        )

        if interval <= 0:
            interval = (
                self.default_maintenance_interval_hours
            )

        age = max(
            float(hours_since_maintenance),
            0.0,
        )

        ratio = (
            age / interval
        )

        if ratio <= 0.50:
            return self._clamp(
                ratio * 0.20
            )

        if ratio <= 0.80:
            return self._scale(
                ratio,
                0.50,
                0.80,
                0.10,
                0.40,
            )

        if ratio <= 1.00:
            return self._scale(
                ratio,
                0.80,
                1.00,
                0.40,
                0.70,
            )

        if ratio <= 1.50:
            return self._scale(
                ratio,
                1.00,
                1.50,
                0.70,
                1.00,
            )

        return 1.0

    # =================================================================
    # Cross-signal correlation
    # =================================================================

    def _cross_signal_adjustment(
        self,
        *,
        risk: float,
        risks: Mapping[str, float],
        degradation_rate: float,
        reasons: list[str],
    ) -> float:

        adjusted = risk

        high_signals = [
            name
            for name, value
            in risks.items()
            if value >= 0.70
        ]

        count = len(
            high_signals
        )

        if count >= 5:

            adjusted += 0.16

            reasons.append(
                "Five or more failure indicators are simultaneously high."
            )

        elif count == 4:

            adjusted += 0.12

            reasons.append(
                "Four failure indicators are simultaneously high."
            )

        elif count == 3:

            adjusted += 0.09

            reasons.append(
                "Three failure indicators are simultaneously high."
            )

        elif count == 2:

            adjusted += 0.05

            reasons.append(
                "Multiple failure indicators are elevated."
            )

        if (
            risks["failure_probability"] >= 0.75
            and risks["degradation"] >= 0.70
        ):

            adjusted += 0.05

            reasons.append(
                "Failure prediction and degradation evidence agree."
            )

        if (
            risks["anomaly"] >= 0.75
            and risks["telemetry"] >= 0.70
        ):

            adjusted += 0.04

            reasons.append(
                "Anomaly and telemetry evidence agree."
            )

        if (
            risks["rul"] >= 0.80
            and risks["degradation"] >= 0.70
        ):

            adjusted += 0.05

            reasons.append(
                "Short RUL agrees with elevated degradation."
            )

        if degradation_rate >= 0.02:

            adjusted += 0.04

            reasons.append(
                "Rapid degradation is increasing failure risk."
            )

        return self._clamp(
            adjusted
        )

    # =================================================================
    # Failure override
    # =================================================================

    def _failure_probability_override(
        self,
        *,
        risk: float,
        failure_probability: float,
        reasons: list[str],
    ) -> float:

        if failure_probability >= 0.95:

            reasons.append(
                "Extreme model-predicted failure probability."
            )

            return max(
                risk,
                0.88,
            )

        if failure_probability >= 0.85:

            reasons.append(
                "Critical model-predicted failure probability."
            )

            return max(
                risk,
                0.78,
            )

        if failure_probability >= 0.75:

            reasons.append(
                "High model-predicted failure probability."
            )

            return max(
                risk,
                0.68,
            )

        return risk

    # =================================================================
    # RUL override
    # =================================================================

    def _rul_override(
        self,
        *,
        risk: float,
        rul_hours: Optional[float],
        reasons: list[str],
    ) -> float:

        if rul_hours is None:
            return risk

        rul = max(
            float(rul_hours),
            0.0,
        )

        if rul <= 24:

            reasons.append(
                "Predicted RUL is 24 hours or less."
            )

            return max(
                risk,
                0.90,
            )

        if rul <= 72:

            reasons.append(
                "Predicted RUL is 72 hours or less."
            )

            return max(
                risk,
                0.82,
            )

        if rul <= 168:

            reasons.append(
                "Predicted RUL is one week or less."
            )

            return max(
                risk,
                0.70,
            )

        return risk

    # =================================================================
    # Degradation-rate adjustment
    # =================================================================

    def _degradation_rate_adjustment(
        self,
        *,
        risk: float,
        degradation_rate: float,
        reasons: list[str],
    ) -> float:

        rate = float(
            degradation_rate
        )

        if rate >= 0.03:

            reasons.append(
                "Extremely rapid degradation trend detected."
            )

            risk += 0.08

        elif rate >= 0.02:

            reasons.append(
                "Rapid degradation trend detected."
            )

            risk += 0.05

        elif rate >= 0.005:

            risk += 0.02

        elif rate <= -0.01:

            risk -= 0.02

        return self._clamp(
            risk
        )

    # =================================================================
    # Utilization
    # =================================================================

    def _utilization_adjustment(
        self,
        *,
        risk: float,
        utilization: float,
    ) -> float:

        utilization = max(
            float(utilization),
            0.0,
        )

        if utilization <= 1.0:
            return risk

        overload = min(
            utilization - 1.0,
            1.0,
        )

        return self._clamp(
            risk
            + 0.08 * overload
        )

    # =================================================================
    # Criticality
    # =================================================================

    def _criticality_adjustment(
        self,
        *,
        risk: float,
        criticality: float,
    ) -> float:

        criticality = self._clamp(
            criticality
        )

        # Criticality modifies operational urgency.
        # It does not claim to change physical failure probability.

        return self._clamp(
            risk
            * (
                1.0
                + 0.10 * criticality
            )
        )

    # =================================================================
    # Temporal smoothing
    # =================================================================

    def _temporal_smoothing(
        self,
        *,
        current: float,
        previous: Optional[float],
    ) -> float:

        if previous is None:
            return current

        previous = self._clamp(
            previous
        )

        # Preserve responsiveness while reducing noisy score jumps.

        return self._clamp(
            0.85 * current
            + 0.15 * previous
        )

    # =================================================================
    # Risk level
    # =================================================================

    @staticmethod
    def _risk_level(
        score: float,
    ) -> FailureRiskLevel:

        if score >= 0.85:
            return FailureRiskLevel.CRITICAL

        if score >= 0.70:
            return FailureRiskLevel.HIGH

        if score >= 0.45:
            return FailureRiskLevel.MEDIUM

        if score >= 0.20:
            return FailureRiskLevel.LOW

        return FailureRiskLevel.NORMAL

    # =================================================================
    # Failure state
    # =================================================================

    @staticmethod
    def _failure_state(
        score: float,
    ) -> FailureRiskState:

        if score >= 0.90:
            return (
                FailureRiskState
                .FAILURE_IMMINENT
            )

        if score >= 0.75:
            return (
                FailureRiskState
                .FAILURE_LIKELY
            )

        if score >= 0.50:
            return (
                FailureRiskState
                .DEGRADING
            )

        if score >= 0.25:
            return FailureRiskState.WATCH

        return FailureRiskState.HEALTHY

    # =================================================================
    # Trend
    # =================================================================

    def _risk_trend(
        self,
        *,
        current: float,
        previous: Optional[float],
    ) -> FailureRiskTrend:

        if previous is None:
            return FailureRiskTrend.STABLE

        previous = self._clamp(
            previous
        )

        delta = (
            current - previous
        )

        if delta >= 0.15:
            return (
                FailureRiskTrend
                .RAPIDLY_INCREASING
            )

        if delta >= 0.03:
            return (
                FailureRiskTrend
                .INCREASING
            )

        if delta <= -0.03:
            return (
                FailureRiskTrend
                .IMPROVING
            )

        return FailureRiskTrend.STABLE

    # =================================================================
    # Confidence
    # =================================================================

    def _overall_confidence(
        self,
        *,
        confidences: Mapping[str, float],
        data: FailureRiskInput,
    ) -> float:

        weighted_confidence = 0.0

        for name, confidence in confidences.items():

            weighted_confidence += (
                self.weights[name]
                * confidence
            )

        evidence_count = sum(
            (
                data.failure_probability > 0,
                data.degradation_score > 0,
                data.rul_hours is not None,
                data.anomaly_score > 0,
                data.telemetry_risk > 0,
                data.hours_since_maintenance > 0,
            )
        )

        coverage = (
            evidence_count / 6.0
        )

        confidence = (
            0.75 * weighted_confidence
            + 0.25 * coverage
        )

        return self._clamp(
            confidence
        )

    # =================================================================
    # Dominant factors
    # =================================================================

    @staticmethod
    def _dominant_factors(
        *,
        risks: Mapping[str, float],
        contributions: Mapping[str, float],
        degradation_rate: float,
    ) -> list[str]:

        ranked = [
            (
                name,
                contributions.get(
                    name,
                    0.0,
                ),
                risk,
            )
            for name, risk
            in risks.items()
        ]

        ranked.sort(
            key=lambda item: (
                item[1],
                item[2],
            ),
            reverse=True,
        )

        factors = [
            name
            for name, contribution, risk
            in ranked
            if contribution > 0 or risk > 0
        ][:5]

        if (
            degradation_rate >= 0.02
            and "rapid_degradation_rate"
            not in factors
        ):
            factors.insert(
                0,
                "rapid_degradation_rate",
            )

        return factors[:5]

    # =================================================================
    # Recommended actions
    # =================================================================

    @staticmethod
    def _recommended_actions(
        *,
        level: FailureRiskLevel,
        state: FailureRiskState,
        trend: FailureRiskTrend,
        dominant_factors: list[str],
    ) -> list[str]:

        actions: list[str] = []

        if level == FailureRiskLevel.CRITICAL:

            actions.extend(
                [
                    "Request immediate engineering inspection.",
                    "Create a critical-priority maintenance work order.",
                    "Validate failure prediction and RUL using independent evidence.",
                    "Increase condition-monitoring frequency.",
                ]
            )

        elif level == FailureRiskLevel.HIGH:

            actions.extend(
                [
                    "Create a priority predictive-maintenance work order.",
                    "Schedule urgent condition-based inspection.",
                    "Review degradation and RUL forecasts.",
                ]
            )

        elif level == FailureRiskLevel.MEDIUM:

            actions.extend(
                [
                    "Schedule inspection within the next maintenance window.",
                    "Continue enhanced condition monitoring.",
                ]
            )

        elif level == FailureRiskLevel.LOW:

            actions.append(
                "Continue enhanced predictive monitoring."
            )

        else:

            actions.append(
                "Continue routine predictive-maintenance monitoring."
            )

        if (
            state
            == FailureRiskState.FAILURE_IMMINENT
        ):
            actions.append(
                "Escalate the failure forecast to the responsible engineering team."
            )

        if (
            trend
            == FailureRiskTrend.RAPIDLY_INCREASING
        ):
            actions.append(
                "Investigate the rapid increase in failure risk."
            )

        if "maintenance" in dominant_factors:
            actions.append(
                "Review maintenance history and overdue maintenance tasks."
            )

        if "anomaly" in dominant_factors:
            actions.append(
                "Investigate recent anomalous telemetry events."
            )

        if "telemetry" in dominant_factors:
            actions.append(
                "Review abnormal sensor measurements and sensor quality."
            )

        return list(
            dict.fromkeys(actions)
        )

    # =================================================================
    # Utilities
    # =================================================================

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
            min(
                1.0,
                ratio,
            ),
        )

        return (
            output_min
            + ratio
            * (
                output_max
                - output_min
            )
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        try:
            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not isfinite(
            value
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )


# =====================================================================
# Factory
# =====================================================================


def create_default_failure_risk_model(
) -> FailureRiskModel:

    return FailureRiskModel(
        failure_probability_weight=0.32,
        degradation_weight=0.22,
        rul_weight=0.18,
        anomaly_weight=0.12,
        telemetry_weight=0.10,
        maintenance_weight=0.06,
        default_maintenance_interval_hours=2000.0,
    )


# =====================================================================
# Convenience API
# =====================================================================


def calculate_failure_risk(
    *,
    asset_id: str,
    failure_probability: float,
    degradation_score: float,
    degradation_rate: float = 0.0,
    rul_hours: Optional[float] = None,
    anomaly_score: float = 0.0,
    telemetry_risk: float = 0.0,
    hours_since_maintenance: float = 0.0,
    expected_maintenance_interval_hours: Optional[float] = None,
    utilization: float = 1.0,
    criticality: float = 0.5,
    previous_risk_score: Optional[float] = None,
) -> Dict[str, Any]:

    model = (
        create_default_failure_risk_model()
    )

    return model.predict_dict(
        asset_id=asset_id,

        failure_probability=(
            failure_probability
        ),

        degradation_score=(
            degradation_score
        ),

        degradation_rate=(
            degradation_rate
        ),

        rul_hours=rul_hours,

        anomaly_score=anomaly_score,
        telemetry_risk=telemetry_risk,

        hours_since_maintenance=(
            hours_since_maintenance
        ),

        expected_maintenance_interval_hours=(
            expected_maintenance_interval_hours
        ),

        utilization=utilization,
        criticality=criticality,

        previous_risk_score=(
            previous_risk_score
        ),
    )


# =====================================================================
# Demo
# =====================================================================


if __name__ == "__main__":

    from pprint import pprint

    model = (
        create_default_failure_risk_model()
    )

    result = model.predict_dict(
        asset_id="PUMP-001",

        failure_probability=0.82,

        degradation_score=0.74,
        degradation_rate=0.021,

        rul_hours=120,

        anomaly_score=0.79,
        telemetry_risk=0.76,

        hours_since_maintenance=1900,
        expected_maintenance_interval_hours=2000,

        utilization=1.15,
        criticality=0.90,

        previous_risk_score=0.58,

        failure_confidence=0.94,
        degradation_confidence=0.91,
        rul_confidence=0.87,
        anomaly_confidence=0.90,
        telemetry_confidence=0.95,
    )

    pprint(result)