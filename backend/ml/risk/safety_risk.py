"""
Industrial Safety Risk Engine
=============================

Aggregates industrial safety evidence into a normalized and explainable
safety-risk assessment.

Typical inputs:
- fire detection
- smoke detection
- overheating detection
- hazardous-condition detection
- unsafe proximity
- restricted-zone violations
- PPE/person safety
- abnormal motion
- telemetry risk
- anomaly risk
- incident correlation
- asset/site criticality

Outputs:
- safety risk score [0, 1]
- safety level/state
- confidence
- dominant hazards
- escalation reasons
- recommended actions

IMPORTANT
---------
This module is decision-support software.

It must NOT replace certified fire protection, emergency shutdown,
SIS/ESD, PLC interlocks, gas detection, alarms, or other validated
industrial safety systems.
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


class SafetyRiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyState(str, Enum):
    SAFE = "safe"
    WATCH = "watch"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


class SafetyTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    INCREASING = "increasing"
    RAPIDLY_INCREASING = "rapidly_increasing"


# =====================================================================
# Input
# =====================================================================


@dataclass(frozen=True)
class SafetyRiskInput:
    asset_id: str

    # -------------------------------------------------------------
    # Primary hazards
    # -------------------------------------------------------------

    fire_risk: float = 0.0
    smoke_risk: float = 0.0
    overheating_risk: float = 0.0

    hazard_risk: float = 0.0

    # -------------------------------------------------------------
    # Worker / zone safety
    # -------------------------------------------------------------

    unsafe_proximity_risk: float = 0.0
    restricted_zone_risk: float = 0.0

    person_risk: float = 0.0
    ppe_risk: float = 0.0
    motion_risk: float = 0.0

    # -------------------------------------------------------------
    # Machine evidence
    # -------------------------------------------------------------

    telemetry_risk: float = 0.0
    anomaly_risk: float = 0.0

    # -------------------------------------------------------------
    # Incident correlation
    # -------------------------------------------------------------

    incident_risk: float = 0.0

    # -------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------

    fire_confidence: float = 1.0
    smoke_confidence: float = 1.0
    overheating_confidence: float = 1.0
    hazard_confidence: float = 1.0

    proximity_confidence: float = 1.0
    restricted_zone_confidence: float = 1.0

    person_confidence: float = 1.0
    ppe_confidence: float = 1.0
    motion_confidence: float = 1.0

    telemetry_confidence: float = 1.0
    anomaly_confidence: float = 1.0
    incident_confidence: float = 1.0

    # -------------------------------------------------------------
    # Context
    # -------------------------------------------------------------

    asset_criticality: float = 0.5
    site_criticality: float = 0.5

    workers_present: bool = False

    previous_risk_score: Optional[float] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# =====================================================================
# Output
# =====================================================================


@dataclass
class SafetyRiskResult:
    asset_id: str
    timestamp: str

    risk_score: float

    risk_level: SafetyRiskLevel
    safety_state: SafetyState
    trend: SafetyTrend

    confidence: float

    fire_risk: float
    smoke_risk: float
    overheating_risk: float
    hazard_risk: float

    unsafe_proximity_risk: float
    restricted_zone_risk: float
    person_risk: float
    ppe_risk: float
    motion_risk: float

    telemetry_risk: float
    anomaly_risk: float
    incident_risk: float

    contributions: Dict[str, float]

    dominant_hazards: list[str]
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

            "safety_state":
                self.safety_state.value,

            "trend":
                self.trend.value,

            "confidence": round(
                self.confidence,
                4,
            ),

            "fire_risk": round(
                self.fire_risk,
                4,
            ),

            "smoke_risk": round(
                self.smoke_risk,
                4,
            ),

            "overheating_risk": round(
                self.overheating_risk,
                4,
            ),

            "hazard_risk": round(
                self.hazard_risk,
                4,
            ),

            "unsafe_proximity_risk": round(
                self.unsafe_proximity_risk,
                4,
            ),

            "restricted_zone_risk": round(
                self.restricted_zone_risk,
                4,
            ),

            "person_risk": round(
                self.person_risk,
                4,
            ),

            "ppe_risk": round(
                self.ppe_risk,
                4,
            ),

            "motion_risk": round(
                self.motion_risk,
                4,
            ),

            "telemetry_risk": round(
                self.telemetry_risk,
                4,
            ),

            "anomaly_risk": round(
                self.anomaly_risk,
                4,
            ),

            "incident_risk": round(
                self.incident_risk,
                4,
            ),

            "contributions": {
                name: round(
                    value,
                    4,
                )
                for name, value
                in self.contributions.items()
            },

            "dominant_hazards":
                self.dominant_hazards,

            "escalation_reasons":
                self.escalation_reasons,

            "recommended_actions":
                self.recommended_actions,
        }


# =====================================================================
# Safety Risk Model
# =====================================================================


class SafetyRiskModel:
    """
    Explainable safety-risk fusion model.

    Critical hazards are intentionally protected from being hidden by
    weighted averaging.

    Example:

        fire risk = 0.98

    must remain critical even if unrelated signals are near zero.
    """

    def __init__(
        self,
        *,
        fire_weight: float = 0.18,
        smoke_weight: float = 0.12,
        overheating_weight: float = 0.11,
        hazard_weight: float = 0.13,
        proximity_weight: float = 0.09,
        restricted_zone_weight: float = 0.07,
        person_weight: float = 0.05,
        ppe_weight: float = 0.05,
        motion_weight: float = 0.04,
        telemetry_weight: float = 0.06,
        anomaly_weight: float = 0.05,
        incident_weight: float = 0.05,
    ) -> None:

        weights = {
            "fire": fire_weight,
            "smoke": smoke_weight,
            "overheating": overheating_weight,
            "hazard": hazard_weight,
            "unsafe_proximity": proximity_weight,
            "restricted_zone": restricted_zone_weight,
            "person": person_weight,
            "ppe": ppe_weight,
            "motion": motion_weight,
            "telemetry": telemetry_weight,
            "anomaly": anomaly_weight,
            "incident": incident_weight,
        }

        if any(
            value < 0
            for value in weights.values()
        ):
            raise ValueError(
                "Safety-risk weights cannot be negative."
            )

        total = sum(
            weights.values()
        )

        if total <= 0:
            raise ValueError(
                "At least one safety-risk weight must be positive."
            )

        self.weights = {
            name: value / total
            for name, value
            in weights.items()
        }

    # =================================================================
    # Main prediction
    # =================================================================

    def predict(
        self,
        data: SafetyRiskInput,
    ) -> SafetyRiskResult:

        risks = {
            "fire":
                self._clamp(
                    data.fire_risk
                ),

            "smoke":
                self._clamp(
                    data.smoke_risk
                ),

            "overheating":
                self._clamp(
                    data.overheating_risk
                ),

            "hazard":
                self._clamp(
                    data.hazard_risk
                ),

            "unsafe_proximity":
                self._clamp(
                    data.unsafe_proximity_risk
                ),

            "restricted_zone":
                self._clamp(
                    data.restricted_zone_risk
                ),

            "person":
                self._clamp(
                    data.person_risk
                ),

            "ppe":
                self._clamp(
                    data.ppe_risk
                ),

            "motion":
                self._clamp(
                    data.motion_risk
                ),

            "telemetry":
                self._clamp(
                    data.telemetry_risk
                ),

            "anomaly":
                self._clamp(
                    data.anomaly_risk
                ),

            "incident":
                self._clamp(
                    data.incident_risk
                ),
        }

        confidences = {
            "fire":
                self._clamp(
                    data.fire_confidence
                ),

            "smoke":
                self._clamp(
                    data.smoke_confidence
                ),

            "overheating":
                self._clamp(
                    data.overheating_confidence
                ),

            "hazard":
                self._clamp(
                    data.hazard_confidence
                ),

            "unsafe_proximity":
                self._clamp(
                    data.proximity_confidence
                ),

            "restricted_zone":
                self._clamp(
                    data.restricted_zone_confidence
                ),

            "person":
                self._clamp(
                    data.person_confidence
                ),

            "ppe":
                self._clamp(
                    data.ppe_confidence
                ),

            "motion":
                self._clamp(
                    data.motion_confidence
                ),

            "telemetry":
                self._clamp(
                    data.telemetry_confidence
                ),

            "anomaly":
                self._clamp(
                    data.anomaly_confidence
                ),

            "incident":
                self._clamp(
                    data.incident_confidence
                ),
        }

        # -------------------------------------------------------------
        # Confidence-aware aggregation
        # -------------------------------------------------------------

        score, contributions = (
            self._aggregate(
                risks=risks,
                confidences=confidences,
            )
        )

        escalation_reasons: list[str] = []

        # -------------------------------------------------------------
        # Cross-signal correlations
        # -------------------------------------------------------------

        score = self._cross_signal_adjustment(
            score=score,
            risks=risks,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Fire protection
        # -------------------------------------------------------------

        score = self._fire_override(
            score=score,
            risks=risks,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Smoke + heat correlation
        # -------------------------------------------------------------

        score = self._thermal_event_override(
            score=score,
            risks=risks,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Worker safety protection
        # -------------------------------------------------------------

        score = self._worker_safety_override(
            score=score,
            risks=risks,
            workers_present=data.workers_present,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Incident correlation
        # -------------------------------------------------------------

        score = self._incident_override(
            score=score,
            incident_risk=risks["incident"],
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Site / asset criticality
        # -------------------------------------------------------------

        score = self._criticality_adjustment(
            score=score,
            asset_criticality=(
                data.asset_criticality
            ),
            site_criticality=(
                data.site_criticality
            ),
        )

        # -------------------------------------------------------------
        # Temporal smoothing
        #
        # Do NOT smooth extreme hazards downward.
        # -------------------------------------------------------------

        critical_signal = max(
            risks["fire"],
            risks["smoke"],
            risks["hazard"],
            risks["unsafe_proximity"],
            risks["incident"],
        )

        if critical_signal < 0.85:

            score = self._temporal_smoothing(
                current=score,
                previous=(
                    data.previous_risk_score
                ),
            )

        score = self._clamp(
            score
        )

        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        level = self._risk_level(
            score
        )

        state = self._safety_state(
            score
        )

        trend = self._trend(
            current=score,
            previous=data.previous_risk_score,
        )

        confidence = self._overall_confidence(
            confidences=confidences,
            risks=risks,
        )

        dominant = self._dominant_hazards(
            risks=risks,
            contributions=contributions,
        )

        actions = self._recommended_actions(
            level=level,
            state=state,
            risks=risks,
            workers_present=data.workers_present,
        )

        return SafetyRiskResult(
            asset_id=data.asset_id,

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            risk_score=score,

            risk_level=level,
            safety_state=state,
            trend=trend,

            confidence=confidence,

            fire_risk=risks["fire"],
            smoke_risk=risks["smoke"],
            overheating_risk=(
                risks["overheating"]
            ),

            hazard_risk=risks["hazard"],

            unsafe_proximity_risk=(
                risks["unsafe_proximity"]
            ),

            restricted_zone_risk=(
                risks["restricted_zone"]
            ),

            person_risk=risks["person"],
            ppe_risk=risks["ppe"],
            motion_risk=risks["motion"],

            telemetry_risk=risks["telemetry"],
            anomaly_risk=risks["anomaly"],
            incident_risk=risks["incident"],

            contributions=contributions,

            dominant_hazards=dominant,

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

        fire_risk: float = 0.0,
        smoke_risk: float = 0.0,
        overheating_risk: float = 0.0,
        hazard_risk: float = 0.0,

        unsafe_proximity_risk: float = 0.0,
        restricted_zone_risk: float = 0.0,

        person_risk: float = 0.0,
        ppe_risk: float = 0.0,
        motion_risk: float = 0.0,

        telemetry_risk: float = 0.0,
        anomaly_risk: float = 0.0,

        incident_risk: float = 0.0,

        workers_present: bool = False,

        asset_criticality: float = 0.5,
        site_criticality: float = 0.5,

        previous_risk_score: Optional[
            float
        ] = None,
    ) -> Dict[str, Any]:

        data = SafetyRiskInput(
            asset_id=asset_id,

            fire_risk=fire_risk,
            smoke_risk=smoke_risk,
            overheating_risk=(
                overheating_risk
            ),

            hazard_risk=hazard_risk,

            unsafe_proximity_risk=(
                unsafe_proximity_risk
            ),

            restricted_zone_risk=(
                restricted_zone_risk
            ),

            person_risk=person_risk,
            ppe_risk=ppe_risk,
            motion_risk=motion_risk,

            telemetry_risk=telemetry_risk,
            anomaly_risk=anomaly_risk,

            incident_risk=incident_risk,

            workers_present=workers_present,

            asset_criticality=(
                asset_criticality
            ),

            site_criticality=(
                site_criticality
            ),

            previous_risk_score=(
                previous_risk_score
            ),
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

            weight = self.weights[
                name
            ]

            confidence = confidences[
                name
            ]

            effective_weight = (
                weight
                * confidence
            )

            contribution = (
                risk
                * effective_weight
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
    # Cross-signal correlation
    # =================================================================

    def _cross_signal_adjustment(
        self,
        *,
        score: float,
        risks: Mapping[str, float],
        reasons: list[str],
    ) -> float:

        high_signals = [
            name
            for name, value
            in risks.items()
            if value >= 0.70
        ]

        count = len(
            high_signals
        )

        if count >= 6:

            score += 0.20

            reasons.append(
                "Six or more safety indicators are simultaneously high."
            )

        elif count >= 4:

            score += 0.15

            reasons.append(
                "Four or more safety indicators are simultaneously high."
            )

        elif count == 3:

            score += 0.10

            reasons.append(
                "Three safety indicators are simultaneously high."
            )

        elif count == 2:

            score += 0.06

            reasons.append(
                "Multiple safety indicators are elevated."
            )

        # Sensor + anomaly agreement.

        if (
            risks["telemetry"] >= 0.75
            and risks["anomaly"] >= 0.75
        ):

            score += 0.04

            reasons.append(
                "Telemetry and anomaly models indicate the same abnormal condition."
            )

        return self._clamp(
            score
        )

    # =================================================================
    # Fire override
    # =================================================================

    def _fire_override(
        self,
        *,
        score: float,
        risks: Mapping[str, float],
        reasons: list[str],
    ) -> float:

        fire = risks[
            "fire"
        ]

        if fire >= 0.95:

            reasons.append(
                "Extreme fire-risk evidence detected."
            )

            return max(
                score,
                0.95,
            )

        if fire >= 0.85:

            reasons.append(
                "Critical fire-risk evidence detected."
            )

            return max(
                score,
                0.88,
            )

        if fire >= 0.70:

            reasons.append(
                "High fire-risk evidence detected."
            )

            return max(
                score,
                0.72,
            )

        return score

    # =================================================================
    # Thermal event correlation
    # =================================================================

    def _thermal_event_override(
        self,
        *,
        score: float,
        risks: Mapping[str, float],
        reasons: list[str],
    ) -> float:

        fire = risks[
            "fire"
        ]

        smoke = risks[
            "smoke"
        ]

        heat = risks[
            "overheating"
        ]

        if (
            fire >= 0.70
            and smoke >= 0.70
            and heat >= 0.70
        ):

            reasons.append(
                "Fire, smoke, and overheating evidence are simultaneously high."
            )

            return max(
                score,
                0.92,
            )

        if (
            smoke >= 0.80
            and heat >= 0.80
        ):

            reasons.append(
                "Smoke and overheating evidence indicate a possible thermal event."
            )

            return max(
                score,
                0.82,
            )

        if (
            fire >= 0.70
            and smoke >= 0.70
        ):

            reasons.append(
                "Fire and smoke evidence are correlated."
            )

            return max(
                score,
                0.85,
            )

        return score

    # =================================================================
    # Worker safety
    # =================================================================

    def _worker_safety_override(
        self,
        *,
        score: float,
        risks: Mapping[str, float],
        workers_present: bool,
        reasons: list[str],
    ) -> float:

        proximity = risks[
            "unsafe_proximity"
        ]

        restricted = risks[
            "restricted_zone"
        ]

        ppe = risks[
            "ppe"
        ]

        if (
            workers_present
            and proximity >= 0.90
        ):

            reasons.append(
                "Extreme unsafe-proximity risk while workers are present."
            )

            score = max(
                score,
                0.90,
            )

        elif (
            workers_present
            and proximity >= 0.75
        ):

            reasons.append(
                "High unsafe-proximity risk while workers are present."
            )

            score = max(
                score,
                0.75,
            )

        if (
            workers_present
            and restricted >= 0.80
        ):

            reasons.append(
                "Restricted-zone violation detected while workers are present."
            )

            score = max(
                score,
                0.78,
            )

        if (
            workers_present
            and ppe >= 0.80
        ):

            reasons.append(
                "High PPE-related safety risk detected."
            )

            score = max(
                score,
                0.70,
            )

        return self._clamp(
            score
        )

    # =================================================================
    # Incident override
    # =================================================================

    def _incident_override(
        self,
        *,
        score: float,
        incident_risk: float,
        reasons: list[str],
    ) -> float:

        if incident_risk >= 0.90:

            reasons.append(
                "Critical correlated incident risk detected."
            )

            return max(
                score,
                0.90,
            )

        if incident_risk >= 0.75:

            reasons.append(
                "High correlated incident risk detected."
            )

            return max(
                score,
                0.75,
            )

        return score

    # =================================================================
    # Criticality
    # =================================================================

    def _criticality_adjustment(
        self,
        *,
        score: float,
        asset_criticality: float,
        site_criticality: float,
    ) -> float:

        asset = self._clamp(
            asset_criticality
        )

        site = self._clamp(
            site_criticality
        )

        combined = (
            0.40 * asset
            + 0.60 * site
        )

        # Criticality affects escalation priority,
        # not the physical existence of a hazard.

        multiplier = (
            1.0
            + 0.08 * combined
        )

        return self._clamp(
            score * multiplier
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

        # Safety stays highly responsive.

        return self._clamp(
            0.90 * current
            + 0.10 * previous
        )

    # =================================================================
    # Risk level
    # =================================================================

    @staticmethod
    def _risk_level(
        score: float,
    ) -> SafetyRiskLevel:

        if score >= 0.85:
            return SafetyRiskLevel.CRITICAL

        if score >= 0.70:
            return SafetyRiskLevel.HIGH

        if score >= 0.45:
            return SafetyRiskLevel.MEDIUM

        if score >= 0.20:
            return SafetyRiskLevel.LOW

        return SafetyRiskLevel.NORMAL

    # =================================================================
    # Safety state
    # =================================================================

    @staticmethod
    def _safety_state(
        score: float,
    ) -> SafetyState:

        if score >= 0.90:
            return SafetyState.EMERGENCY

        if score >= 0.75:
            return SafetyState.DANGER

        if score >= 0.50:
            return SafetyState.WARNING

        if score >= 0.25:
            return SafetyState.WATCH

        return SafetyState.SAFE

    # =================================================================
    # Trend
    # =================================================================

    def _trend(
        self,
        *,
        current: float,
        previous: Optional[float],
    ) -> SafetyTrend:

        if previous is None:
            return SafetyTrend.STABLE

        previous = self._clamp(
            previous
        )

        change = (
            current
            - previous
        )

        if change >= 0.15:
            return (
                SafetyTrend
                .RAPIDLY_INCREASING
            )

        if change >= 0.03:
            return SafetyTrend.INCREASING

        if change <= -0.03:
            return SafetyTrend.IMPROVING

        return SafetyTrend.STABLE

    # =================================================================
    # Confidence
    # =================================================================

    def _overall_confidence(
        self,
        *,
        confidences: Mapping[str, float],
        risks: Mapping[str, float],
    ) -> float:

        weighted_confidence = 0.0

        for name, confidence in confidences.items():

            weighted_confidence += (
                self.weights[name]
                * confidence
            )

        active_evidence = sum(
            value > 0
            for value in risks.values()
        )

        coverage = min(
            active_evidence / 6.0,
            1.0,
        )

        confidence = (
            0.80 * weighted_confidence
            + 0.20 * coverage
        )

        return self._clamp(
            confidence
        )

    # =================================================================
    # Dominant hazards
    # =================================================================

    @staticmethod
    def _dominant_hazards(
        *,
        risks: Mapping[str, float],
        contributions: Mapping[str, float],
    ) -> list[str]:

        ranking = [
            (
                name,
                risk,
                contributions.get(
                    name,
                    0.0,
                ),
            )
            for name, risk
            in risks.items()
        ]

        # Safety severity is primary.
        # Weighted contribution is secondary.

        ranking.sort(
            key=lambda item: (
                item[1],
                item[2],
            ),
            reverse=True,
        )

        return [
            name
            for name, risk, contribution
            in ranking
            if (
                risk > 0
                or contribution > 0
            )
        ][:6]

    # =================================================================
    # Recommended actions
    # =================================================================

    @staticmethod
    def _recommended_actions(
        *,
        level: SafetyRiskLevel,
        state: SafetyState,
        risks: Mapping[str, float],
        workers_present: bool,
    ) -> list[str]:

        actions: list[str] = []

        if level == SafetyRiskLevel.CRITICAL:

            actions.extend(
                [
                    "Escalate immediately to the responsible safety team.",
                    "Validate the hazard using independent sensors and safety systems.",
                    "Open a critical-priority safety incident.",
                ]
            )

        elif level == SafetyRiskLevel.HIGH:

            actions.extend(
                [
                    "Open a high-priority safety review.",
                    "Increase safety monitoring frequency.",
                    "Inspect the affected equipment or zone.",
                ]
            )

        elif level == SafetyRiskLevel.MEDIUM:

            actions.extend(
                [
                    "Schedule a safety inspection.",
                    "Review the dominant hazard signals.",
                ]
            )

        elif level == SafetyRiskLevel.LOW:

            actions.append(
                "Continue enhanced safety monitoring."
            )

        else:

            actions.append(
                "Continue routine safety monitoring."
            )

        if (
            risks["fire"] >= 0.70
            or risks["smoke"] >= 0.70
        ):

            actions.append(
                "Verify fire/smoke evidence against certified detection and alarm systems."
            )

        if risks["overheating"] >= 0.70:

            actions.append(
                "Inspect thermal condition and recent temperature telemetry."
            )

        if (
            workers_present
            and risks["unsafe_proximity"] >= 0.70
        ):

            actions.append(
                "Escalate unsafe worker-equipment proximity for immediate site review."
            )

        if (
            workers_present
            and risks["restricted_zone"] >= 0.70
        ):

            actions.append(
                "Review the restricted-zone access event."
            )

        if (
            workers_present
            and risks["ppe"] >= 0.70
        ):

            actions.append(
                "Review PPE-compliance evidence."
            )

        if risks["hazard"] >= 0.70:

            actions.append(
                "Review the detected hazard and applicable site safety procedure."
            )

        if risks["incident"] >= 0.70:

            actions.append(
                "Correlate the event with active and recent safety incidents."
            )

        return list(
            dict.fromkeys(
                actions
            )
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


def create_default_safety_risk_model(
) -> SafetyRiskModel:

    return SafetyRiskModel(
        fire_weight=0.18,
        smoke_weight=0.12,
        overheating_weight=0.11,
        hazard_weight=0.13,

        proximity_weight=0.09,
        restricted_zone_weight=0.07,

        person_weight=0.05,
        ppe_weight=0.05,
        motion_weight=0.04,

        telemetry_weight=0.06,
        anomaly_weight=0.05,

        incident_weight=0.05,
    )


# =====================================================================
# Convenience API
# =====================================================================


def calculate_safety_risk(
    *,
    asset_id: str,

    fire_risk: float = 0.0,
    smoke_risk: float = 0.0,
    overheating_risk: float = 0.0,
    hazard_risk: float = 0.0,

    unsafe_proximity_risk: float = 0.0,
    restricted_zone_risk: float = 0.0,

    person_risk: float = 0.0,
    ppe_risk: float = 0.0,
    motion_risk: float = 0.0,

    telemetry_risk: float = 0.0,
    anomaly_risk: float = 0.0,

    incident_risk: float = 0.0,

    workers_present: bool = False,

    asset_criticality: float = 0.5,
    site_criticality: float = 0.5,

    previous_risk_score: Optional[
        float
    ] = None,
) -> Dict[str, Any]:

    model = (
        create_default_safety_risk_model()
    )

    return model.predict_dict(
        asset_id=asset_id,

        fire_risk=fire_risk,
        smoke_risk=smoke_risk,
        overheating_risk=overheating_risk,

        hazard_risk=hazard_risk,

        unsafe_proximity_risk=(
            unsafe_proximity_risk
        ),

        restricted_zone_risk=(
            restricted_zone_risk
        ),

        person_risk=person_risk,
        ppe_risk=ppe_risk,
        motion_risk=motion_risk,

        telemetry_risk=telemetry_risk,
        anomaly_risk=anomaly_risk,

        incident_risk=incident_risk,

        workers_present=workers_present,

        asset_criticality=(
            asset_criticality
        ),

        site_criticality=(
            site_criticality
        ),

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
        create_default_safety_risk_model()
    )

    result = model.predict_dict(
        asset_id="FURNACE-001",

        fire_risk=0.83,
        smoke_risk=0.79,
        overheating_risk=0.88,

        hazard_risk=0.72,

        unsafe_proximity_risk=0.67,
        restricted_zone_risk=0.45,

        person_risk=0.35,
        ppe_risk=0.52,
        motion_risk=0.28,

        telemetry_risk=0.82,
        anomaly_risk=0.77,

        incident_risk=0.73,

        workers_present=True,

        asset_criticality=0.95,
        site_criticality=0.90,

        previous_risk_score=0.61,
    )

    pprint(
        result
    )