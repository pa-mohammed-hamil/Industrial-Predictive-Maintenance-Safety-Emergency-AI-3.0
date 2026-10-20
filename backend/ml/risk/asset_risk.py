"""
Asset Risk Engine
=================

Calculates the operational risk of an industrial asset using:

- asset health
- failure probability
- RUL
- degradation
- anomaly detection
- telemetry condition
- maintenance condition
- visual inspection
- asset criticality
- utilization

Output:
- normalized asset risk score [0, 1]
- risk level
- health score
- dominant risk factors
- confidence
- recommended actions

This module performs decision-support risk scoring. Safety-critical
physical actions must be handled by validated safety/control systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from math import exp, isfinite
from typing import Any, Dict, Mapping, Optional


# =====================================================================
# Enums
# =====================================================================


class AssetRiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssetHealthState(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


# =====================================================================
# Input
# =====================================================================


@dataclass(frozen=True)
class AssetRiskInput:
    asset_id: str

    # Predictive maintenance
    failure_probability: float = 0.0

    degradation_score: float = 0.0
    degradation_rate: float = 0.0

    rul_hours: Optional[float] = None

    # Anomaly layer
    anomaly_score: float = 0.0

    # Telemetry
    telemetry_risk: float = 0.0

    # Vision / inspection
    visual_risk: float = 0.0

    # Maintenance
    hours_since_maintenance: float = 0.0
    expected_maintenance_interval_hours: Optional[float] = None

    # Business/asset context
    criticality: float = 0.5
    utilization: float = 1.0

    # Optional externally calculated health score
    health_score: Optional[float] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# =====================================================================
# Output
# =====================================================================


@dataclass
class AssetRiskResult:
    asset_id: str
    timestamp: str

    risk_score: float
    risk_level: AssetRiskLevel

    health_score: float
    health_state: AssetHealthState

    failure_risk: float
    degradation_risk: float
    rul_risk: float
    anomaly_risk: float
    telemetry_risk: float
    visual_risk: float
    maintenance_risk: float

    confidence: float

    dominant_factors: list[str]
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

            "health_score": round(
                self.health_score,
                4,
            ),

            "health_state":
                self.health_state.value,

            "failure_risk": round(
                self.failure_risk,
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

            "visual_risk": round(
                self.visual_risk,
                4,
            ),

            "maintenance_risk": round(
                self.maintenance_risk,
                4,
            ),

            "confidence": round(
                self.confidence,
                4,
            ),

            "dominant_factors":
                self.dominant_factors,

            "recommended_actions":
                self.recommended_actions,
        }


# =====================================================================
# Main engine
# =====================================================================


class AssetRiskModel:
    """
    Explainable asset-level risk aggregation engine.

    Combines ML/CV/telemetry/maintenance evidence into one asset score.

    The weights are configurable because different industrial asset
    classes have different reliability and safety characteristics.
    """

    def __init__(
        self,
        *,
        failure_weight: float = 0.25,
        degradation_weight: float = 0.18,
        rul_weight: float = 0.15,
        anomaly_weight: float = 0.12,
        telemetry_weight: float = 0.12,
        visual_weight: float = 0.10,
        maintenance_weight: float = 0.08,
        default_maintenance_interval_hours: float = 2000.0,
    ) -> None:

        weights = {
            "failure": failure_weight,
            "degradation": degradation_weight,
            "rul": rul_weight,
            "anomaly": anomaly_weight,
            "telemetry": telemetry_weight,
            "visual": visual_weight,
            "maintenance": maintenance_weight,
        }

        if any(
            value < 0
            for value in weights.values()
        ):
            raise ValueError(
                "Risk weights cannot be negative."
            )

        total = sum(
            weights.values()
        )

        if total <= 0:
            raise ValueError(
                "At least one risk weight must be positive."
            )

        self.weights = {
            name: value / total
            for name, value in weights.items()
        }

        if default_maintenance_interval_hours <= 0:
            raise ValueError(
                "Maintenance interval must be positive."
            )

        self.default_maintenance_interval_hours = (
            float(
                default_maintenance_interval_hours
            )
        )

    # =================================================================
    # Public API
    # =================================================================

    def predict(
        self,
        data: AssetRiskInput,
    ) -> AssetRiskResult:

        failure_risk = self._clamp(
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

        visual_risk = self._clamp(
            data.visual_risk
        )

        rul_risk = self._rul_risk(
            data.rul_hours
        )

        maintenance_risk = (
            self._maintenance_risk(
                hours_since_maintenance=(
                    data.hours_since_maintenance
                ),
                expected_interval=(
                    data.expected_maintenance_interval_hours
                ),
            )
        )

        # -------------------------------------------------------------
        # Weighted base risk
        # -------------------------------------------------------------

        base_risk = (
            self.weights["failure"]
            * failure_risk

            + self.weights["degradation"]
            * degradation_risk

            + self.weights["rul"]
            * rul_risk

            + self.weights["anomaly"]
            * anomaly_risk

            + self.weights["telemetry"]
            * telemetry_risk

            + self.weights["visual"]
            * visual_risk

            + self.weights["maintenance"]
            * maintenance_risk
        )

        # -------------------------------------------------------------
        # Cross-signal evidence
        # -------------------------------------------------------------

        risk_score = (
            self._cross_signal_adjustment(
                base_risk=base_risk,
                failure_risk=failure_risk,
                degradation_risk=degradation_risk,
                rul_risk=rul_risk,
                anomaly_risk=anomaly_risk,
                telemetry_risk=telemetry_risk,
                visual_risk=visual_risk,
            )
        )

        # -------------------------------------------------------------
        # Asset context
        # -------------------------------------------------------------

        risk_score = (
            self._criticality_adjustment(
                risk_score,
                data.criticality,
            )
        )

        risk_score = (
            self._utilization_adjustment(
                risk_score,
                data.utilization,
            )
        )

        risk_score = self._clamp(
            risk_score
        )

        # -------------------------------------------------------------
        # Health
        # -------------------------------------------------------------

        if data.health_score is not None:

            health_score = self._clamp(
                data.health_score
            )

            # External health assessment can reinforce risk.
            risk_score = max(
                risk_score,
                1.0 - health_score,
            )

        else:

            health_score = self._clamp(
                1.0 - risk_score
            )

        risk_level = self._risk_level(
            risk_score
        )

        health_state = self._health_state(
            health_score
        )

        # -------------------------------------------------------------
        # Explainability
        # -------------------------------------------------------------

        factors = self._dominant_factors(
            failure_risk=failure_risk,
            degradation_risk=degradation_risk,
            rul_risk=rul_risk,
            anomaly_risk=anomaly_risk,
            telemetry_risk=telemetry_risk,
            visual_risk=visual_risk,
            maintenance_risk=maintenance_risk,
        )

        confidence = self._confidence(
            data
        )

        actions = self._recommended_actions(
            risk_level=risk_level,
            factors=factors,
        )

        return AssetRiskResult(
            asset_id=data.asset_id,

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            risk_score=risk_score,
            risk_level=risk_level,

            health_score=health_score,
            health_state=health_state,

            failure_risk=failure_risk,
            degradation_risk=degradation_risk,
            rul_risk=rul_risk,
            anomaly_risk=anomaly_risk,
            telemetry_risk=telemetry_risk,
            visual_risk=visual_risk,
            maintenance_risk=maintenance_risk,

            confidence=confidence,

            dominant_factors=factors,
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
        visual_risk: float = 0.0,
        hours_since_maintenance: float = 0.0,
        expected_maintenance_interval_hours: Optional[
            float
        ] = None,
        criticality: float = 0.5,
        utilization: float = 1.0,
        health_score: Optional[float] = None,
    ) -> Dict[str, Any]:

        data = AssetRiskInput(
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

            visual_risk=visual_risk,

            hours_since_maintenance=(
                hours_since_maintenance
            ),

            expected_maintenance_interval_hours=(
                expected_maintenance_interval_hours
            ),

            criticality=criticality,
            utilization=utilization,

            health_score=health_score,
        )

        return self.predict(
            data
        ).to_dict()

    # =================================================================
    # RUL risk
    # =================================================================

    @staticmethod
    def _rul_risk(
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
            return 0.90

        if rul <= 168:
            return 0.75

        if rul <= 336:
            return 0.60

        if rul <= 720:
            return 0.40

        if rul <= 1440:
            return 0.20

        return 0.05

    # =================================================================
    # Maintenance risk
    # =================================================================

    def _maintenance_risk(
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

        if ratio <= 0:
            return 0.0

        if ratio <= 0.50:
            return 0.20 * ratio

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
        base_risk: float,
        failure_risk: float,
        degradation_risk: float,
        rul_risk: float,
        anomaly_risk: float,
        telemetry_risk: float,
        visual_risk: float,
    ) -> float:

        risk = base_risk

        high_signals = sum(
            score >= 0.70
            for score in (
                failure_risk,
                degradation_risk,
                rul_risk,
                anomaly_risk,
                telemetry_risk,
                visual_risk,
            )
        )

        if high_signals >= 5:
            risk += 0.18

        elif high_signals >= 4:
            risk += 0.14

        elif high_signals >= 3:
            risk += 0.10

        elif high_signals >= 2:
            risk += 0.06

        # Failure + degradation agreement.
        if (
            failure_risk >= 0.80
            and degradation_risk >= 0.80
        ):
            risk += 0.05

        # Telemetry + anomaly agreement.
        if (
            telemetry_risk >= 0.80
            and anomaly_risk >= 0.80
        ):
            risk += 0.04

        # Visual + telemetry agreement.
        if (
            visual_risk >= 0.80
            and telemetry_risk >= 0.80
        ):
            risk += 0.05

        # Extremely short RUL should never be hidden by averaging.
        if rul_risk >= 0.90:
            risk = max(
                risk,
                0.70,
            )

        return self._clamp(
            risk
        )

    # =================================================================
    # Criticality
    # =================================================================

    def _criticality_adjustment(
        self,
        risk: float,
        criticality: float,
    ) -> float:

        criticality = self._clamp(
            criticality
        )

        multiplier = (
            1.0
            + 0.15 * criticality
        )

        return self._clamp(
            risk * multiplier
        )

    # =================================================================
    # Utilization
    # =================================================================

    def _utilization_adjustment(
        self,
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
    # Risk classification
    # =================================================================

    @staticmethod
    def _risk_level(
        risk: float,
    ) -> AssetRiskLevel:

        if risk >= 0.85:
            return AssetRiskLevel.CRITICAL

        if risk >= 0.70:
            return AssetRiskLevel.HIGH

        if risk >= 0.45:
            return AssetRiskLevel.MEDIUM

        if risk >= 0.20:
            return AssetRiskLevel.LOW

        return AssetRiskLevel.NORMAL

    # =================================================================
    # Health classification
    # =================================================================

    @staticmethod
    def _health_state(
        health: float,
    ) -> AssetHealthState:

        if health >= 0.85:
            return AssetHealthState.EXCELLENT

        if health >= 0.70:
            return AssetHealthState.GOOD

        if health >= 0.50:
            return AssetHealthState.FAIR

        if health >= 0.25:
            return AssetHealthState.POOR

        return AssetHealthState.CRITICAL

    # =================================================================
    # Confidence
    # =================================================================

    @staticmethod
    def _confidence(
        data: AssetRiskInput,
    ) -> float:

        evidence = 0
        total = 7

        if data.failure_probability > 0:
            evidence += 1

        if data.degradation_score > 0:
            evidence += 1

        if data.rul_hours is not None:
            evidence += 1

        if data.anomaly_score > 0:
            evidence += 1

        if data.telemetry_risk > 0:
            evidence += 1

        if data.visual_risk > 0:
            evidence += 1

        if data.hours_since_maintenance > 0:
            evidence += 1

        coverage = (
            evidence / total
        )

        return min(
            1.0,
            0.30
            + 0.70 * coverage,
        )

    # =================================================================
    # Explainability
    # =================================================================

    def _dominant_factors(
        self,
        *,
        failure_risk: float,
        degradation_risk: float,
        rul_risk: float,
        anomaly_risk: float,
        telemetry_risk: float,
        visual_risk: float,
        maintenance_risk: float,
    ) -> list[str]:

        factors = {
            "failure_probability":
                failure_risk
                * self.weights["failure"],

            "degradation":
                degradation_risk
                * self.weights["degradation"],

            "remaining_useful_life":
                rul_risk
                * self.weights["rul"],

            "anomaly_detection":
                anomaly_risk
                * self.weights["anomaly"],

            "telemetry_condition":
                telemetry_risk
                * self.weights["telemetry"],

            "visual_condition":
                visual_risk
                * self.weights["visual"],

            "maintenance_age":
                maintenance_risk
                * self.weights["maintenance"],
        }

        ranked = sorted(
            factors.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            name
            for name, value in ranked
            if value > 0
        ][:5]

    # =================================================================
    # Recommended actions
    # =================================================================

    @staticmethod
    def _recommended_actions(
        *,
        risk_level: AssetRiskLevel,
        factors: list[str],
    ) -> list[str]:

        actions: list[str] = []

        if risk_level == AssetRiskLevel.CRITICAL:

            actions.extend(
                [
                    "Request immediate engineering inspection.",
                    "Create a critical-priority maintenance work order.",
                    "Validate failure, RUL, anomaly, and telemetry evidence.",
                ]
            )

        elif risk_level == AssetRiskLevel.HIGH:

            actions.extend(
                [
                    "Create a priority maintenance work order.",
                    "Schedule condition-based inspection.",
                    "Increase telemetry monitoring frequency.",
                ]
            )

        elif risk_level == AssetRiskLevel.MEDIUM:

            actions.extend(
                [
                    "Schedule inspection within the next maintenance window.",
                    "Review degradation and RUL trends.",
                ]
            )

        elif risk_level == AssetRiskLevel.LOW:

            actions.append(
                "Continue enhanced condition monitoring."
            )

        else:

            actions.append(
                "Continue routine asset monitoring."
            )

        if "visual_condition" in factors:
            actions.append(
                "Review recent visual inspection evidence."
            )

        if "remaining_useful_life" in factors:
            actions.append(
                "Review the latest RUL prediction before scheduling maintenance."
            )

        if "maintenance_age" in factors:
            actions.append(
                "Review preventive-maintenance history and due dates."
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

        value = float(
            value
        )

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


def create_default_asset_risk_model(
) -> AssetRiskModel:

    return AssetRiskModel(
        failure_weight=0.25,
        degradation_weight=0.18,
        rul_weight=0.15,
        anomaly_weight=0.12,
        telemetry_weight=0.12,
        visual_weight=0.10,
        maintenance_weight=0.08,
        default_maintenance_interval_hours=2000.0,
    )


# =====================================================================
# Convenience API
# =====================================================================


def calculate_asset_risk(
    *,
    asset_id: str,
    failure_probability: float = 0.0,
    degradation_score: float = 0.0,
    degradation_rate: float = 0.0,
    rul_hours: Optional[float] = None,
    anomaly_score: float = 0.0,
    telemetry_risk: float = 0.0,
    visual_risk: float = 0.0,
    hours_since_maintenance: float = 0.0,
    expected_maintenance_interval_hours: Optional[float] = None,
    criticality: float = 0.5,
    utilization: float = 1.0,
) -> Dict[str, Any]:

    model = (
        create_default_asset_risk_model()
    )

    return model.predict_dict(
        asset_id=asset_id,
        failure_probability=failure_probability,
        degradation_score=degradation_score,
        degradation_rate=degradation_rate,
        rul_hours=rul_hours,
        anomaly_score=anomaly_score,
        telemetry_risk=telemetry_risk,
        visual_risk=visual_risk,
        hours_since_maintenance=hours_since_maintenance,
        expected_maintenance_interval_hours=(
            expected_maintenance_interval_hours
        ),
        criticality=criticality,
        utilization=utilization,
    )


# =====================================================================
# Local demonstration
# =====================================================================


if __name__ == "__main__":

    model = create_default_asset_risk_model()

    result = model.predict_dict(
        asset_id="MOTOR-001",

        failure_probability=0.78,

        degradation_score=0.72,
        degradation_rate=0.018,

        rul_hours=185,

        anomaly_score=0.81,

        telemetry_risk=0.76,
        visual_risk=0.55,

        hours_since_maintenance=1850,
        expected_maintenance_interval_hours=2000,

        criticality=0.90,
        utilization=1.12,
    )

    from pprint import pprint

    pprint(result)