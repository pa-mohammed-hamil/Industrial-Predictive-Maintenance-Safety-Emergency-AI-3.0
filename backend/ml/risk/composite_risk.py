"""
Composite Industrial Risk Engine
================================

Combines multiple risk domains into one enterprise/asset-level risk score.

Risk domains:
- Asset risk
- Failure risk
- Safety risk
- Financial risk
- Optional operational / environmental risk

Features:
- Configurable normalized weights
- Critical-risk escalation
- Cross-domain correlation bonuses
- Confidence-aware aggregation
- Risk level classification
- Dominant factor analysis
- Risk contribution breakdown
- Recommended actions
- Explainable JSON output

This module provides decision-support risk aggregation.
Safety-critical actions must remain under validated industrial safety
and control systems.
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


class CompositeRiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CompositeRiskState(str, Enum):
    STABLE = "stable"
    WATCH = "watch"
    ELEVATED = "elevated"
    SEVERE = "severe"
    EMERGENCY = "emergency"


class RiskTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    INCREASING = "increasing"
    RAPIDLY_INCREASING = "rapidly_increasing"


# =====================================================================
# Input model
# =====================================================================


@dataclass(frozen=True)
class CompositeRiskInput:
    asset_id: str

    asset_risk: float = 0.0
    failure_risk: float = 0.0
    safety_risk: float = 0.0
    financial_risk: float = 0.0

    operational_risk: float = 0.0
    environmental_risk: float = 0.0

    # Confidence from each upstream model.
    asset_confidence: float = 1.0
    failure_confidence: float = 1.0
    safety_confidence: float = 1.0
    financial_confidence: float = 1.0
    operational_confidence: float = 1.0
    environmental_confidence: float = 1.0

    # Asset/business context.
    asset_criticality: float = 0.5
    business_criticality: float = 0.5

    # Optional previous score for trend analysis.
    previous_risk_score: Optional[float] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# =====================================================================
# Output model
# =====================================================================


@dataclass
class CompositeRiskResult:
    asset_id: str
    timestamp: str

    risk_score: float
    risk_level: CompositeRiskLevel
    risk_state: CompositeRiskState
    trend: RiskTrend

    confidence: float

    asset_risk: float
    failure_risk: float
    safety_risk: float
    financial_risk: float
    operational_risk: float
    environmental_risk: float

    contributions: Dict[str, float]

    dominant_risks: list[str]
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

            "risk_level": self.risk_level.value,
            "risk_state": self.risk_state.value,
            "trend": self.trend.value,

            "confidence": round(
                self.confidence,
                4,
            ),

            "asset_risk": round(
                self.asset_risk,
                4,
            ),

            "failure_risk": round(
                self.failure_risk,
                4,
            ),

            "safety_risk": round(
                self.safety_risk,
                4,
            ),

            "financial_risk": round(
                self.financial_risk,
                4,
            ),

            "operational_risk": round(
                self.operational_risk,
                4,
            ),

            "environmental_risk": round(
                self.environmental_risk,
                4,
            ),

            "contributions": {
                key: round(value, 4)
                for key, value
                in self.contributions.items()
            },

            "dominant_risks":
                self.dominant_risks,

            "escalation_reasons":
                self.escalation_reasons,

            "recommended_actions":
                self.recommended_actions,
        }


# =====================================================================
# Composite risk engine
# =====================================================================


class CompositeRiskModel:
    """
    Top-level industrial risk aggregation engine.

    Upstream models should independently calculate:

        asset_risk.py
        failure_risk.py
        safety_risk.py
        financial_risk.py

    This model fuses those outputs instead of recalculating them.
    """

    def __init__(
        self,
        *,
        asset_weight: float = 0.22,
        failure_weight: float = 0.23,
        safety_weight: float = 0.30,
        financial_weight: float = 0.12,
        operational_weight: float = 0.08,
        environmental_weight: float = 0.05,
        critical_threshold: float = 0.85,
    ) -> None:

        weights = {
            "asset_risk": asset_weight,
            "failure_risk": failure_weight,
            "safety_risk": safety_weight,
            "financial_risk": financial_weight,
            "operational_risk": operational_weight,
            "environmental_risk": environmental_weight,
        }

        if any(
            weight < 0
            for weight in weights.values()
        ):
            raise ValueError(
                "Composite-risk weights cannot be negative."
            )

        total = sum(
            weights.values()
        )

        if total <= 0:
            raise ValueError(
                "At least one risk weight must be positive."
            )

        self.weights = {
            name: weight / total
            for name, weight in weights.items()
        }

        if not 0 < critical_threshold <= 1:
            raise ValueError(
                "critical_threshold must be within (0, 1]."
            )

        self.critical_threshold = float(
            critical_threshold
        )

    # =================================================================
    # Main prediction
    # =================================================================

    def predict(
        self,
        data: CompositeRiskInput,
    ) -> CompositeRiskResult:

        risks = {
            "asset_risk": self._clamp(
                data.asset_risk
            ),

            "failure_risk": self._clamp(
                data.failure_risk
            ),

            "safety_risk": self._clamp(
                data.safety_risk
            ),

            "financial_risk": self._clamp(
                data.financial_risk
            ),

            "operational_risk": self._clamp(
                data.operational_risk
            ),

            "environmental_risk": self._clamp(
                data.environmental_risk
            ),
        }

        confidences = {
            "asset_risk": self._clamp(
                data.asset_confidence
            ),

            "failure_risk": self._clamp(
                data.failure_confidence
            ),

            "safety_risk": self._clamp(
                data.safety_confidence
            ),

            "financial_risk": self._clamp(
                data.financial_confidence
            ),

            "operational_risk": self._clamp(
                data.operational_confidence
            ),

            "environmental_risk": self._clamp(
                data.environmental_confidence
            ),
        }

        # -------------------------------------------------------------
        # Confidence-aware weighted aggregation
        # -------------------------------------------------------------

        base_score, contributions = (
            self._aggregate(
                risks=risks,
                confidences=confidences,
            )
        )

        escalation_reasons: list[str] = []

        # -------------------------------------------------------------
        # Cross-domain correlation
        # -------------------------------------------------------------

        score = self._apply_cross_domain_rules(
            base_score=base_score,
            risks=risks,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Critical safety override
        # -------------------------------------------------------------

        score = self._apply_safety_override(
            score=score,
            safety_risk=risks["safety_risk"],
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Failure escalation
        # -------------------------------------------------------------

        score = self._apply_failure_override(
            score=score,
            failure_risk=risks["failure_risk"],
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Asset / business criticality
        # -------------------------------------------------------------

        score = self._apply_criticality(
            score=score,
            asset_criticality=data.asset_criticality,
            business_criticality=data.business_criticality,
        )

        score = self._clamp(
            score
        )

        # -------------------------------------------------------------
        # Overall confidence
        # -------------------------------------------------------------

        confidence = self._overall_confidence(
            confidences
        )

        # -------------------------------------------------------------
        # Risk classification
        # -------------------------------------------------------------

        level = self._risk_level(
            score
        )

        state = self._risk_state(
            score
        )

        trend = self._risk_trend(
            current=score,
            previous=data.previous_risk_score,
        )

        # -------------------------------------------------------------
        # Explainability
        # -------------------------------------------------------------

        dominant = self._dominant_risks(
            risks=risks,
            contributions=contributions,
        )

        actions = self._recommended_actions(
            level=level,
            risks=risks,
            trend=trend,
        )

        return CompositeRiskResult(
            asset_id=data.asset_id,

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            risk_score=score,
            risk_level=level,
            risk_state=state,
            trend=trend,

            confidence=confidence,

            asset_risk=risks["asset_risk"],
            failure_risk=risks["failure_risk"],
            safety_risk=risks["safety_risk"],
            financial_risk=risks["financial_risk"],
            operational_risk=risks["operational_risk"],
            environmental_risk=risks[
                "environmental_risk"
            ],

            contributions=contributions,

            dominant_risks=dominant,

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
        asset_risk: float = 0.0,
        failure_risk: float = 0.0,
        safety_risk: float = 0.0,
        financial_risk: float = 0.0,
        operational_risk: float = 0.0,
        environmental_risk: float = 0.0,
        asset_confidence: float = 1.0,
        failure_confidence: float = 1.0,
        safety_confidence: float = 1.0,
        financial_confidence: float = 1.0,
        operational_confidence: float = 1.0,
        environmental_confidence: float = 1.0,
        asset_criticality: float = 0.5,
        business_criticality: float = 0.5,
        previous_risk_score: Optional[float] = None,
    ) -> Dict[str, Any]:

        data = CompositeRiskInput(
            asset_id=asset_id,

            asset_risk=asset_risk,
            failure_risk=failure_risk,
            safety_risk=safety_risk,
            financial_risk=financial_risk,

            operational_risk=operational_risk,
            environmental_risk=environmental_risk,

            asset_confidence=asset_confidence,
            failure_confidence=failure_confidence,
            safety_confidence=safety_confidence,
            financial_confidence=financial_confidence,
            operational_confidence=(
                operational_confidence
            ),
            environmental_confidence=(
                environmental_confidence
            ),

            asset_criticality=asset_criticality,
            business_criticality=business_criticality,

            previous_risk_score=previous_risk_score,
        )

        return self.predict(
            data
        ).to_dict()

    # =================================================================
    # Aggregation
    # =================================================================

    def _aggregate(
        self,
        *,
        risks: Mapping[str, float],
        confidences: Mapping[str, float],
    ) -> tuple[float, Dict[str, float]]:

        weighted_total = 0.0
        effective_weight_total = 0.0

        raw: Dict[str, float] = {}

        for name, risk in risks.items():

            base_weight = self.weights[
                name
            ]

            confidence = confidences[
                name
            ]

            effective_weight = (
                base_weight * confidence
            )

            contribution = (
                risk * effective_weight
            )

            raw[name] = contribution

            weighted_total += contribution

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
            in raw.items()
        }

        return (
            self._clamp(score),
            contributions,
        )

    # =================================================================
    # Cross-domain correlation
    # =================================================================

    def _apply_cross_domain_rules(
        self,
        *,
        base_score: float,
        risks: Mapping[str, float],
        reasons: list[str],
    ) -> float:

        score = base_score

        high_domains = [
            name
            for name, value in risks.items()
            if value >= 0.70
        ]

        count = len(
            high_domains
        )

        if count >= 5:

            score += 0.18

            reasons.append(
                "Five or more risk domains are simultaneously high."
            )

        elif count == 4:

            score += 0.14

            reasons.append(
                "Four risk domains are simultaneously high."
            )

        elif count == 3:

            score += 0.10

            reasons.append(
                "Three risk domains are simultaneously high."
            )

        elif count == 2:

            score += 0.06

            reasons.append(
                "Multiple risk domains are simultaneously elevated."
            )

        # Failure + asset condition.
        if (
            risks["failure_risk"] >= 0.75
            and risks["asset_risk"] >= 0.70
        ):

            score += 0.05

            reasons.append(
                "Failure risk and asset-condition risk agree."
            )

        # Failure + financial exposure.
        if (
            risks["failure_risk"] >= 0.75
            and risks["financial_risk"] >= 0.70
        ):

            score += 0.04

            reasons.append(
                "High failure probability coincides with high financial exposure."
            )

        # Safety + operational.
        if (
            risks["safety_risk"] >= 0.75
            and risks["operational_risk"] >= 0.70
        ):

            score += 0.06

            reasons.append(
                "Safety and operational risks are simultaneously elevated."
            )

        return self._clamp(
            score
        )

    # =================================================================
    # Safety escalation
    # =================================================================

    def _apply_safety_override(
        self,
        *,
        score: float,
        safety_risk: float,
        reasons: list[str],
    ) -> float:

        # Composite averaging must not hide severe safety evidence.

        if safety_risk >= 0.95:

            reasons.append(
                "Extreme safety risk triggered critical escalation."
            )

            return max(
                score,
                0.90,
            )

        if safety_risk >= 0.85:

            reasons.append(
                "Critical safety risk triggered escalation."
            )

            return max(
                score,
                0.85,
            )

        if safety_risk >= 0.75:

            reasons.append(
                "High safety risk raised the composite risk floor."
            )

            return max(
                score,
                0.70,
            )

        return score

    # =================================================================
    # Failure escalation
    # =================================================================

    def _apply_failure_override(
        self,
        *,
        score: float,
        failure_risk: float,
        reasons: list[str],
    ) -> float:

        if failure_risk >= 0.95:

            reasons.append(
                "Extreme failure risk triggered escalation."
            )

            return max(
                score,
                0.85,
            )

        if failure_risk >= 0.85:

            reasons.append(
                "Critical failure risk raised the composite risk floor."
            )

            return max(
                score,
                0.75,
            )

        return score

    # =================================================================
    # Criticality
    # =================================================================

    def _apply_criticality(
        self,
        *,
        score: float,
        asset_criticality: float,
        business_criticality: float,
    ) -> float:

        asset = self._clamp(
            asset_criticality
        )

        business = self._clamp(
            business_criticality
        )

        combined = (
            0.60 * asset
            + 0.40 * business
        )

        # Maximum uplift = 10%.
        multiplier = (
            1.0
            + 0.10 * combined
        )

        return self._clamp(
            score * multiplier
        )

    # =================================================================
    # Confidence
    # =================================================================

    def _overall_confidence(
        self,
        confidences: Mapping[str, float],
    ) -> float:

        weighted = 0.0

        for name, confidence in confidences.items():

            weighted += (
                self.weights[name]
                * confidence
            )

        return self._clamp(
            weighted
        )

    # =================================================================
    # Level
    # =================================================================

    @staticmethod
    def _risk_level(
        score: float,
    ) -> CompositeRiskLevel:

        if score >= 0.85:
            return CompositeRiskLevel.CRITICAL

        if score >= 0.70:
            return CompositeRiskLevel.HIGH

        if score >= 0.45:
            return CompositeRiskLevel.MEDIUM

        if score >= 0.20:
            return CompositeRiskLevel.LOW

        return CompositeRiskLevel.NORMAL

    # =================================================================
    # State
    # =================================================================

    @staticmethod
    def _risk_state(
        score: float,
    ) -> CompositeRiskState:

        if score >= 0.90:
            return CompositeRiskState.EMERGENCY

        if score >= 0.75:
            return CompositeRiskState.SEVERE

        if score >= 0.50:
            return CompositeRiskState.ELEVATED

        if score >= 0.25:
            return CompositeRiskState.WATCH

        return CompositeRiskState.STABLE

    # =================================================================
    # Trend
    # =================================================================

    def _risk_trend(
        self,
        *,
        current: float,
        previous: Optional[float],
    ) -> RiskTrend:

        if previous is None:
            return RiskTrend.STABLE

        previous = self._clamp(
            previous
        )

        change = (
            current - previous
        )

        if change >= 0.15:
            return (
                RiskTrend
                .RAPIDLY_INCREASING
            )

        if change >= 0.03:
            return RiskTrend.INCREASING

        if change <= -0.03:
            return RiskTrend.IMPROVING

        return RiskTrend.STABLE

    # =================================================================
    # Dominant risks
    # =================================================================

    @staticmethod
    def _dominant_risks(
        *,
        risks: Mapping[str, float],
        contributions: Mapping[str, float],
    ) -> list[str]:

        ranking = []

        for name in risks:

            # Contribution is primary, absolute severity secondary.
            ranking.append(
                (
                    name,
                    contributions.get(
                        name,
                        0.0,
                    ),
                    risks[name],
                )
            )

        ranking.sort(
            key=lambda item: (
                item[1],
                item[2],
            ),
            reverse=True,
        )

        return [
            name
            for name, contribution, severity
            in ranking
            if (
                contribution > 0
                or severity > 0
            )
        ][:5]

    # =================================================================
    # Actions
    # =================================================================

    @staticmethod
    def _recommended_actions(
        *,
        level: CompositeRiskLevel,
        risks: Mapping[str, float],
        trend: RiskTrend,
    ) -> list[str]:

        actions: list[str] = []

        if level == CompositeRiskLevel.CRITICAL:

            actions.extend(
                [
                    "Initiate immediate multidisciplinary engineering review.",
                    "Create a critical-priority incident or maintenance case.",
                    "Validate all high-risk signals using independent evidence.",
                ]
            )

        elif level == CompositeRiskLevel.HIGH:

            actions.extend(
                [
                    "Create a priority risk-mitigation work order.",
                    "Schedule urgent engineering inspection.",
                    "Increase monitoring frequency.",
                ]
            )

        elif level == CompositeRiskLevel.MEDIUM:

            actions.extend(
                [
                    "Schedule condition-based inspection.",
                    "Review the dominant risk domains.",
                ]
            )

        elif level == CompositeRiskLevel.LOW:

            actions.append(
                "Continue enhanced monitoring."
            )

        else:

            actions.append(
                "Continue routine monitoring."
            )

        if risks["safety_risk"] >= 0.70:

            actions.append(
                "Escalate safety evidence to the safety review workflow."
            )

        if risks["failure_risk"] >= 0.70:

            actions.append(
                "Review failure prediction, degradation, and RUL evidence."
            )

        if risks["financial_risk"] >= 0.70:

            actions.append(
                "Review downtime, repair, replacement, and production-loss exposure."
            )

        if risks["asset_risk"] >= 0.70:

            actions.append(
                "Review asset health and maintenance condition."
            )

        if (
            trend
            == RiskTrend.RAPIDLY_INCREASING
        ):

            actions.append(
                "Investigate the rapid increase in composite risk."
            )

        return list(
            dict.fromkeys(actions)
        )

    # =================================================================
    # Utility
    # =================================================================

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


def create_default_composite_risk_model(
) -> CompositeRiskModel:

    return CompositeRiskModel(
        asset_weight=0.22,
        failure_weight=0.23,

        # Safety intentionally has the largest weight.
        safety_weight=0.30,

        financial_weight=0.12,
        operational_weight=0.08,
        environmental_weight=0.05,

        critical_threshold=0.85,
    )


# =====================================================================
# Convenience API
# =====================================================================


def calculate_composite_risk(
    *,
    asset_id: str,
    asset_risk: float,
    failure_risk: float,
    safety_risk: float,
    financial_risk: float,
    operational_risk: float = 0.0,
    environmental_risk: float = 0.0,
    previous_risk_score: Optional[float] = None,
) -> Dict[str, Any]:

    model = (
        create_default_composite_risk_model()
    )

    return model.predict_dict(
        asset_id=asset_id,

        asset_risk=asset_risk,
        failure_risk=failure_risk,
        safety_risk=safety_risk,
        financial_risk=financial_risk,

        operational_risk=operational_risk,
        environmental_risk=environmental_risk,

        previous_risk_score=previous_risk_score,
    )


# =====================================================================
# Demo
# =====================================================================


if __name__ == "__main__":

    from pprint import pprint

    model = (
        create_default_composite_risk_model()
    )

    result = model.predict_dict(
        asset_id="COMPRESSOR-001",

        asset_risk=0.76,
        failure_risk=0.82,

        # High safety evidence should not be hidden by averaging.
        safety_risk=0.88,

        financial_risk=0.69,

        operational_risk=0.72,
        environmental_risk=0.35,

        asset_confidence=0.94,
        failure_confidence=0.91,
        safety_confidence=0.96,
        financial_confidence=0.82,
        operational_confidence=0.85,
        environmental_confidence=0.70,

        asset_criticality=0.95,
        business_criticality=0.90,

        previous_risk_score=0.61,
    )

    pprint(
        result
    )