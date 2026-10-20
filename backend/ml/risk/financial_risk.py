"""
Industrial Financial Risk Engine
================================

Calculates financial risk associated with industrial asset degradation,
failure, maintenance, downtime, production loss, emergency events,
repair/replacement decisions, and expected future failure costs.

Typical upstream inputs
-----------------------
- failure probability
- repair cost
- replacement cost
- downtime cost
- production loss
- spare-parts cost
- labor cost
- emergency cost
- energy-loss cost
- scrap cost
- expected failure cost
- asset value
- revenue / production value
- insurance/recovery offsets

Outputs
-------
- normalized financial risk score [0, 1]
- risk level
- total gross exposure
- recoverable amount
- net exposure
- expected loss
- exposure-to-asset-value ratio
- dominant financial drivers
- confidence
- escalation reasons
- recommended actions

This module is a planning and decision-support component. Financial
thresholds should be configured using the organization's actual asset,
production, insurance, accounting, and maintenance data.
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


class FinancialRiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FinancialExposureState(str, Enum):
    MINIMAL = "minimal"
    MANAGEABLE = "manageable"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    EXTREME = "extreme"


class FinancialRiskTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    INCREASING = "increasing"
    RAPIDLY_INCREASING = "rapidly_increasing"


# =====================================================================
# Input
# =====================================================================


@dataclass(frozen=True)
class FinancialRiskInput:
    asset_id: str

    # -------------------------------------------------------------
    # Predictive-maintenance evidence
    # -------------------------------------------------------------

    failure_probability: float = 0.0
    degradation_score: float = 0.0

    rul_hours: Optional[float] = None

    # -------------------------------------------------------------
    # Direct intervention costs
    # -------------------------------------------------------------

    repair_cost: float = 0.0
    replacement_cost: float = 0.0

    spare_parts_cost: float = 0.0
    labor_cost: float = 0.0

    # -------------------------------------------------------------
    # Operational loss
    # -------------------------------------------------------------

    downtime_cost: float = 0.0
    production_loss: float = 0.0

    energy_loss_cost: float = 0.0
    scrap_cost: float = 0.0

    # -------------------------------------------------------------
    # Incident / emergency exposure
    # -------------------------------------------------------------

    emergency_cost: float = 0.0

    # Optional cost already calculated by another cost model.
    expected_failure_cost: Optional[float] = None

    # -------------------------------------------------------------
    # Financial context
    # -------------------------------------------------------------

    asset_value: float = 0.0

    annual_revenue_impact: float = 0.0

    # Insurance / warranty / vendor recovery.
    recoverable_amount: float = 0.0

    # Optional budget used for budget-impact scoring.
    maintenance_budget: Optional[float] = None

    # -------------------------------------------------------------
    # Context
    # -------------------------------------------------------------

    asset_criticality: float = 0.5
    business_criticality: float = 0.5

    # Previous final financial-risk score.
    previous_risk_score: Optional[float] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# =====================================================================
# Output
# =====================================================================


@dataclass
class FinancialRiskResult:
    asset_id: str
    timestamp: str

    risk_score: float
    risk_level: FinancialRiskLevel
    exposure_state: FinancialExposureState
    trend: FinancialRiskTrend

    confidence: float

    gross_exposure: float
    recoverable_amount: float
    net_exposure: float
    expected_loss: float

    intervention_cost: float
    operational_loss: float
    emergency_exposure: float

    exposure_to_asset_value: Optional[float]
    expected_loss_to_asset_value: Optional[float]
    budget_impact_ratio: Optional[float]

    failure_probability: float
    degradation_score: float
    rul_risk: float

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

            "exposure_state":
                self.exposure_state.value,

            "trend":
                self.trend.value,

            "confidence": round(
                self.confidence,
                4,
            ),

            "gross_exposure": round(
                self.gross_exposure,
                2,
            ),

            "recoverable_amount": round(
                self.recoverable_amount,
                2,
            ),

            "net_exposure": round(
                self.net_exposure,
                2,
            ),

            "expected_loss": round(
                self.expected_loss,
                2,
            ),

            "intervention_cost": round(
                self.intervention_cost,
                2,
            ),

            "operational_loss": round(
                self.operational_loss,
                2,
            ),

            "emergency_exposure": round(
                self.emergency_exposure,
                2,
            ),

            "exposure_to_asset_value": (
                round(
                    self.exposure_to_asset_value,
                    4,
                )
                if self.exposure_to_asset_value is not None
                else None
            ),

            "expected_loss_to_asset_value": (
                round(
                    self.expected_loss_to_asset_value,
                    4,
                )
                if self.expected_loss_to_asset_value is not None
                else None
            ),

            "budget_impact_ratio": (
                round(
                    self.budget_impact_ratio,
                    4,
                )
                if self.budget_impact_ratio is not None
                else None
            ),

            "failure_probability": round(
                self.failure_probability,
                4,
            ),

            "degradation_score": round(
                self.degradation_score,
                4,
            ),

            "rul_risk": round(
                self.rul_risk,
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
# Financial Risk Model
# =====================================================================


class FinancialRiskModel:
    """
    Explainable industrial financial-risk model.

    It evaluates four main dimensions:

        1. expected monetary loss
        2. exposure relative to asset value
        3. operational / production loss
        4. budget impact

    Predictive-maintenance evidence modifies the financial urgency.
    """

    def __init__(
        self,
        *,
        expected_loss_weight: float = 0.35,
        asset_exposure_weight: float = 0.25,
        operational_loss_weight: float = 0.20,
        budget_impact_weight: float = 0.10,
        predictive_risk_weight: float = 0.10,
        default_reference_value: float = 100_000.0,
    ) -> None:

        weights = {
            "expected_loss":
                expected_loss_weight,

            "asset_exposure":
                asset_exposure_weight,

            "operational_loss":
                operational_loss_weight,

            "budget_impact":
                budget_impact_weight,

            "predictive_risk":
                predictive_risk_weight,
        }

        if any(
            value < 0
            for value in weights.values()
        ):
            raise ValueError(
                "Financial-risk weights cannot be negative."
            )

        total = sum(
            weights.values()
        )

        if total <= 0:
            raise ValueError(
                "At least one financial-risk weight must be positive."
            )

        self.weights = {
            name: value / total
            for name, value
            in weights.items()
        }

        if default_reference_value <= 0:
            raise ValueError(
                "default_reference_value must be positive."
            )

        self.default_reference_value = float(
            default_reference_value
        )

    # =================================================================
    # Main prediction
    # =================================================================

    def predict(
        self,
        data: FinancialRiskInput,
    ) -> FinancialRiskResult:

        failure_probability = self._clamp(
            data.failure_probability
        )

        degradation_score = self._clamp(
            data.degradation_score
        )

        rul_risk = self._rul_risk(
            data.rul_hours
        )

        # -------------------------------------------------------------
        # Normalize monetary inputs
        # -------------------------------------------------------------

        repair_cost = self._money(
            data.repair_cost
        )

        replacement_cost = self._money(
            data.replacement_cost
        )

        spare_parts_cost = self._money(
            data.spare_parts_cost
        )

        labor_cost = self._money(
            data.labor_cost
        )

        downtime_cost = self._money(
            data.downtime_cost
        )

        production_loss = self._money(
            data.production_loss
        )

        energy_loss_cost = self._money(
            data.energy_loss_cost
        )

        scrap_cost = self._money(
            data.scrap_cost
        )

        emergency_cost = self._money(
            data.emergency_cost
        )

        asset_value = self._money(
            data.asset_value
        )

        recoverable = self._money(
            data.recoverable_amount
        )

        # -------------------------------------------------------------
        # Intervention cost
        # -------------------------------------------------------------

        repair_intervention = (
            repair_cost
            + spare_parts_cost
            + labor_cost
        )

        if replacement_cost > 0 and repair_intervention > 0:

            # Financial exposure uses the economically smaller planned
            # intervention as the baseline intervention requirement.
            intervention_cost = min(
                repair_intervention,
                replacement_cost,
            )

        elif replacement_cost > 0:

            intervention_cost = replacement_cost

        else:

            intervention_cost = repair_intervention

        # -------------------------------------------------------------
        # Operational losses
        # -------------------------------------------------------------

        operational_loss = (
            downtime_cost
            + production_loss
            + energy_loss_cost
            + scrap_cost
        )

        emergency_exposure = (
            emergency_cost
        )

        # -------------------------------------------------------------
        # Gross / net exposure
        # -------------------------------------------------------------

        gross_exposure = (
            intervention_cost
            + operational_loss
            + emergency_exposure
        )

        net_exposure = max(
            0.0,
            gross_exposure - recoverable,
        )

        # -------------------------------------------------------------
        # Expected loss
        # -------------------------------------------------------------

        if data.expected_failure_cost is not None:

            expected_loss = self._money(
                data.expected_failure_cost
            )

            # Do not allow recoveries to make loss negative.
            expected_loss = max(
                0.0,
                expected_loss - recoverable,
            )

        else:

            expected_loss = self._expected_loss(
                net_exposure=net_exposure,
                failure_probability=failure_probability,
                degradation_score=degradation_score,
                rul_risk=rul_risk,
            )

        # -------------------------------------------------------------
        # Exposure ratios
        # -------------------------------------------------------------

        exposure_to_asset_value = (
            self._safe_ratio(
                net_exposure,
                asset_value,
            )
            if asset_value > 0
            else None
        )

        expected_loss_to_asset_value = (
            self._safe_ratio(
                expected_loss,
                asset_value,
            )
            if asset_value > 0
            else None
        )

        budget = (
            self._money(
                data.maintenance_budget
            )
            if data.maintenance_budget is not None
            else 0.0
        )

        budget_impact_ratio = (
            self._safe_ratio(
                intervention_cost,
                budget,
            )
            if budget > 0
            else None
        )

        # -------------------------------------------------------------
        # Reference value
        # -------------------------------------------------------------

        reference_value = (
            asset_value
            if asset_value > 0
            else self.default_reference_value
        )

        # -------------------------------------------------------------
        # Convert monetary exposure into normalized risks
        # -------------------------------------------------------------

        expected_loss_risk = (
            self._ratio_to_risk(
                expected_loss
                / reference_value
            )
        )

        asset_exposure_risk = (
            self._ratio_to_risk(
                net_exposure
                / reference_value
            )
        )

        operational_loss_risk = (
            self._ratio_to_risk(
                operational_loss
                / reference_value
            )
        )

        if budget_impact_ratio is not None:

            budget_risk = (
                self._budget_ratio_to_risk(
                    budget_impact_ratio
                )
            )

        else:

            budget_risk = 0.0

        predictive_risk = (
            0.50 * failure_probability
            + 0.30 * degradation_score
            + 0.20 * rul_risk
        )

        predictive_risk = self._clamp(
            predictive_risk
        )

        risks = {
            "expected_loss":
                expected_loss_risk,

            "asset_exposure":
                asset_exposure_risk,

            "operational_loss":
                operational_loss_risk,

            "budget_impact":
                budget_risk,

            "predictive_risk":
                predictive_risk,
        }

        # -------------------------------------------------------------
        # Weighted aggregation
        # -------------------------------------------------------------

        score, contributions = (
            self._aggregate(
                risks
            )
        )

        escalation_reasons: list[str] = []

        # -------------------------------------------------------------
        # Cross-risk correlation
        # -------------------------------------------------------------

        score = self._cross_signal_adjustment(
            score=score,
            risks=risks,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Catastrophic financial exposure floor
        # -------------------------------------------------------------

        score = self._exposure_override(
            score=score,
            exposure_ratio=exposure_to_asset_value,
            expected_loss_ratio=(
                expected_loss_to_asset_value
            ),
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Emergency-cost escalation
        # -------------------------------------------------------------

        score = self._emergency_adjustment(
            score=score,
            emergency_cost=emergency_exposure,
            reference_value=reference_value,
            reasons=escalation_reasons,
        )

        # -------------------------------------------------------------
        # Asset/business criticality
        # -------------------------------------------------------------

        score = self._criticality_adjustment(
            score=score,
            asset_criticality=(
                data.asset_criticality
            ),
            business_criticality=(
                data.business_criticality
            ),
        )

        score = self._clamp(
            score
        )

        # -------------------------------------------------------------
        # Trend
        # -------------------------------------------------------------

        trend = self._trend(
            current=score,
            previous=data.previous_risk_score,
        )

        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        level = self._risk_level(
            score
        )

        state = self._exposure_state(
            score
        )

        # -------------------------------------------------------------
        # Confidence
        # -------------------------------------------------------------

        confidence = self._confidence(
            data=data,
            asset_value=asset_value,
        )

        # -------------------------------------------------------------
        # Explainability
        # -------------------------------------------------------------

        dominant = self._dominant_factors(
            risks=risks,
            contributions=contributions,
            monetary_factors={
                "repair_cost": repair_cost,
                "replacement_cost": replacement_cost,
                "downtime_cost": downtime_cost,
                "production_loss": production_loss,
                "emergency_cost": emergency_cost,
                "energy_loss_cost": energy_loss_cost,
                "scrap_cost": scrap_cost,
            },
        )

        actions = self._recommended_actions(
            level=level,
            dominant_factors=dominant,
            repair_cost=repair_intervention,
            replacement_cost=replacement_cost,
            operational_loss=operational_loss,
            expected_loss=expected_loss,
        )

        return FinancialRiskResult(
            asset_id=data.asset_id,

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            risk_score=score,
            risk_level=level,
            exposure_state=state,
            trend=trend,

            confidence=confidence,

            gross_exposure=gross_exposure,
            recoverable_amount=recoverable,
            net_exposure=net_exposure,
            expected_loss=expected_loss,

            intervention_cost=intervention_cost,
            operational_loss=operational_loss,
            emergency_exposure=emergency_exposure,

            exposure_to_asset_value=(
                exposure_to_asset_value
            ),

            expected_loss_to_asset_value=(
                expected_loss_to_asset_value
            ),

            budget_impact_ratio=(
                budget_impact_ratio
            ),

            failure_probability=(
                failure_probability
            ),

            degradation_score=(
                degradation_score
            ),

            rul_risk=rul_risk,

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
        rul_hours: Optional[float] = None,
        repair_cost: float = 0.0,
        replacement_cost: float = 0.0,
        spare_parts_cost: float = 0.0,
        labor_cost: float = 0.0,
        downtime_cost: float = 0.0,
        production_loss: float = 0.0,
        energy_loss_cost: float = 0.0,
        scrap_cost: float = 0.0,
        emergency_cost: float = 0.0,
        expected_failure_cost: Optional[
            float
        ] = None,
        asset_value: float = 0.0,
        annual_revenue_impact: float = 0.0,
        recoverable_amount: float = 0.0,
        maintenance_budget: Optional[
            float
        ] = None,
        asset_criticality: float = 0.5,
        business_criticality: float = 0.5,
        previous_risk_score: Optional[
            float
        ] = None,
    ) -> Dict[str, Any]:

        data = FinancialRiskInput(
            asset_id=asset_id,

            failure_probability=(
                failure_probability
            ),

            degradation_score=(
                degradation_score
            ),

            rul_hours=rul_hours,

            repair_cost=repair_cost,
            replacement_cost=replacement_cost,

            spare_parts_cost=spare_parts_cost,
            labor_cost=labor_cost,

            downtime_cost=downtime_cost,
            production_loss=production_loss,

            energy_loss_cost=energy_loss_cost,
            scrap_cost=scrap_cost,

            emergency_cost=emergency_cost,

            expected_failure_cost=(
                expected_failure_cost
            ),

            asset_value=asset_value,

            annual_revenue_impact=(
                annual_revenue_impact
            ),

            recoverable_amount=(
                recoverable_amount
            ),

            maintenance_budget=(
                maintenance_budget
            ),

            asset_criticality=(
                asset_criticality
            ),

            business_criticality=(
                business_criticality
            ),

            previous_risk_score=(
                previous_risk_score
            ),
        )

        return self.predict(
            data
        ).to_dict()

    # =================================================================
    # Expected loss
    # =================================================================

    def _expected_loss(
        self,
        *,
        net_exposure: float,
        failure_probability: float,
        degradation_score: float,
        rul_risk: float,
    ) -> float:

        # Probability component is primary.
        probability = (
            0.65 * failure_probability
            + 0.20 * degradation_score
            + 0.15 * rul_risk
        )

        probability = self._clamp(
            probability
        )

        return (
            net_exposure
            * probability
        )

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
            return 0.78

        if rul <= 336:
            return 0.60

        if rul <= 720:
            return 0.40

        if rul <= 1440:
            return 0.20

        return 0.05

    # =================================================================
    # Risk aggregation
    # =================================================================

    def _aggregate(
        self,
        risks: Mapping[str, float],
    ) -> tuple[float, Dict[str, float]]:

        contributions: Dict[
            str,
            float
        ] = {}

        score = 0.0

        for name, risk in risks.items():

            contribution = (
                self.weights[name]
                * self._clamp(risk)
            )

            contributions[
                name
            ] = contribution

            score += contribution

        return (
            self._clamp(score),
            contributions,
        )

    # =================================================================
    # Cross-signal adjustment
    # =================================================================

    def _cross_signal_adjustment(
        self,
        *,
        score: float,
        risks: Mapping[str, float],
        reasons: list[str],
    ) -> float:

        high = [
            name
            for name, value
            in risks.items()
            if value >= 0.70
        ]

        if len(high) >= 4:

            score += 0.14

            reasons.append(
                "Four or more financial risk dimensions are simultaneously high."
            )

        elif len(high) == 3:

            score += 0.10

            reasons.append(
                "Three financial risk dimensions are simultaneously high."
            )

        elif len(high) == 2:

            score += 0.06

            reasons.append(
                "Multiple financial risk dimensions are elevated."
            )

        if (
            risks["expected_loss"] >= 0.70
            and risks["predictive_risk"] >= 0.70
        ):

            score += 0.05

            reasons.append(
                "High expected loss agrees with high predicted equipment risk."
            )

        if (
            risks["operational_loss"] >= 0.70
            and risks["asset_exposure"] >= 0.70
        ):

            score += 0.04

            reasons.append(
                "Operational losses and asset exposure are both high."
            )

        return self._clamp(
            score
        )

    # =================================================================
    # Exposure override
    # =================================================================

    def _exposure_override(
        self,
        *,
        score: float,
        exposure_ratio: Optional[float],
        expected_loss_ratio: Optional[float],
        reasons: list[str],
    ) -> float:

        if exposure_ratio is not None:

            if exposure_ratio >= 1.0:

                score = max(
                    score,
                    0.88,
                )

                reasons.append(
                    "Net financial exposure equals or exceeds asset value."
                )

            elif exposure_ratio >= 0.75:

                score = max(
                    score,
                    0.78,
                )

                reasons.append(
                    "Net financial exposure exceeds 75% of asset value."
                )

            elif exposure_ratio >= 0.50:

                score = max(
                    score,
                    0.68,
                )

                reasons.append(
                    "Net financial exposure exceeds 50% of asset value."
                )

        if expected_loss_ratio is not None:

            if expected_loss_ratio >= 0.75:

                score = max(
                    score,
                    0.85,
                )

                reasons.append(
                    "Expected loss is extremely high relative to asset value."
                )

            elif expected_loss_ratio >= 0.40:

                score = max(
                    score,
                    0.70,
                )

                reasons.append(
                    "Expected loss is high relative to asset value."
                )

        return self._clamp(
            score
        )

    # =================================================================
    # Emergency cost
    # =================================================================

    def _emergency_adjustment(
        self,
        *,
        score: float,
        emergency_cost: float,
        reference_value: float,
        reasons: list[str],
    ) -> float:

        if reference_value <= 0:
            return score

        ratio = (
            emergency_cost
            / reference_value
        )

        if ratio >= 0.50:

            score += 0.08

            reasons.append(
                "Potential emergency cost is extremely high."
            )

        elif ratio >= 0.25:

            score += 0.05

            reasons.append(
                "Potential emergency cost is significant."
            )

        return self._clamp(
            score
        )

    # =================================================================
    # Criticality
    # =================================================================

    def _criticality_adjustment(
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
            0.40 * asset
            + 0.60 * business
        )

        # Business criticality affects the severity of financial
        # exposure rather than changing the actual monetary values.

        return self._clamp(
            score
            * (
                1.0
                + 0.12 * combined
            )
        )

    # =================================================================
    # Trend
    # =================================================================

    def _trend(
        self,
        *,
        current: float,
        previous: Optional[float],
    ) -> FinancialRiskTrend:

        if previous is None:
            return FinancialRiskTrend.STABLE

        previous = self._clamp(
            previous
        )

        delta = (
            current - previous
        )

        if delta >= 0.15:
            return (
                FinancialRiskTrend
                .RAPIDLY_INCREASING
            )

        if delta >= 0.03:
            return (
                FinancialRiskTrend
                .INCREASING
            )

        if delta <= -0.03:
            return (
                FinancialRiskTrend
                .IMPROVING
            )

        return FinancialRiskTrend.STABLE

    # =================================================================
    # Risk level
    # =================================================================

    @staticmethod
    def _risk_level(
        score: float,
    ) -> FinancialRiskLevel:

        if score >= 0.85:
            return FinancialRiskLevel.CRITICAL

        if score >= 0.70:
            return FinancialRiskLevel.HIGH

        if score >= 0.45:
            return FinancialRiskLevel.MEDIUM

        if score >= 0.20:
            return FinancialRiskLevel.LOW

        return FinancialRiskLevel.NORMAL

    # =================================================================
    # Exposure state
    # =================================================================

    @staticmethod
    def _exposure_state(
        score: float,
    ) -> FinancialExposureState:

        if score >= 0.85:
            return FinancialExposureState.EXTREME

        if score >= 0.70:
            return FinancialExposureState.SEVERE

        if score >= 0.45:
            return FinancialExposureState.SIGNIFICANT

        if score >= 0.20:
            return FinancialExposureState.MANAGEABLE

        return FinancialExposureState.MINIMAL

    # =================================================================
    # Confidence
    # =================================================================

    def _confidence(
        self,
        *,
        data: FinancialRiskInput,
        asset_value: float,
    ) -> float:

        evidence = 0
        total = 9

        if data.failure_probability > 0:
            evidence += 1

        if data.degradation_score > 0:
            evidence += 1

        if data.rul_hours is not None:
            evidence += 1

        if data.repair_cost > 0:
            evidence += 1

        if data.replacement_cost > 0:
            evidence += 1

        if data.downtime_cost > 0:
            evidence += 1

        if data.production_loss > 0:
            evidence += 1

        if asset_value > 0:
            evidence += 1

        if (
            data.expected_failure_cost
            is not None
        ):
            evidence += 1

        coverage = (
            evidence / total
        )

        return self._clamp(
            0.30
            + 0.70 * coverage
        )

    # =================================================================
    # Dominant factors
    # =================================================================

    @staticmethod
    def _dominant_factors(
        *,
        risks: Mapping[str, float],
        contributions: Mapping[str, float],
        monetary_factors: Mapping[str, float],
    ) -> list[str]:

        ranking = [
            (
                name,
                contributions.get(
                    name,
                    0.0,
                ),
            )
            for name in risks
        ]

        ranking.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        result = [
            name
            for name, value
            in ranking
            if value > 0
        ][:3]

        monetary = sorted(
            monetary_factors.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for name, value in monetary:

            if (
                value > 0
                and name not in result
            ):
                result.append(
                    name
                )

            if len(result) >= 5:
                break

        return result[:5]

    # =================================================================
    # Recommended actions
    # =================================================================

    @staticmethod
    def _recommended_actions(
        *,
        level: FinancialRiskLevel,
        dominant_factors: list[str],
        repair_cost: float,
        replacement_cost: float,
        operational_loss: float,
        expected_loss: float,
    ) -> list[str]:

        actions: list[str] = []

        if level == FinancialRiskLevel.CRITICAL:

            actions.extend(
                [
                    "Escalate financial exposure for immediate engineering and operations review.",
                    "Review repair-versus-replacement economics.",
                    "Prioritize actions that reduce expected downtime and production loss.",
                ]
            )

        elif level == FinancialRiskLevel.HIGH:

            actions.extend(
                [
                    "Review maintenance intervention cost and expected failure loss.",
                    "Evaluate repair-versus-replacement alternatives.",
                    "Prioritize the asset in maintenance planning.",
                ]
            )

        elif level == FinancialRiskLevel.MEDIUM:

            actions.extend(
                [
                    "Review projected maintenance and downtime costs.",
                    "Include the asset in the next financial maintenance review.",
                ]
            )

        elif level == FinancialRiskLevel.LOW:

            actions.append(
                "Continue monitoring financial exposure."
            )

        else:

            actions.append(
                "Continue routine cost monitoring."
            )

        if (
            replacement_cost > 0
            and repair_cost > 0
        ):

            if repair_cost >= (
                replacement_cost * 0.70
            ):

                actions.append(
                    "Perform detailed repair-versus-replacement analysis because repair cost is approaching replacement cost."
                )

        if operational_loss > repair_cost:

            actions.append(
                "Prioritize downtime reduction because operational losses exceed direct repair cost."
            )

        if expected_loss > 0:

            actions.append(
                "Track expected failure cost against the approved maintenance budget."
            )

        if "production_loss" in dominant_factors:

            actions.append(
                "Review production-loss mitigation and scheduling options."
            )

        if "emergency_cost" in dominant_factors:

            actions.append(
                "Review emergency-response financial exposure."
            )

        return list(
            dict.fromkeys(actions)
        )

    # =================================================================
    # Risk transforms
    # =================================================================

    @staticmethod
    def _ratio_to_risk(
        ratio: float,
    ) -> float:

        """
        Convert exposure/reference ratio into risk.

        Ratio     Approx risk
        ---------------------
        0.05      low
        0.25      moderate
        0.50      high
        1.00+     critical
        """

        ratio = max(
            float(ratio),
            0.0,
        )

        if ratio >= 1.0:
            return 1.0

        if ratio >= 0.75:
            return 0.90

        if ratio >= 0.50:
            return 0.75

        if ratio >= 0.25:
            return 0.55

        if ratio >= 0.10:
            return 0.35

        if ratio >= 0.05:
            return 0.20

        return min(
            ratio * 4.0,
            0.20,
        )

    @staticmethod
    def _budget_ratio_to_risk(
        ratio: float,
    ) -> float:

        ratio = max(
            float(ratio),
            0.0,
        )

        if ratio >= 1.0:
            return 1.0

        if ratio >= 0.75:
            return 0.85

        if ratio >= 0.50:
            return 0.65

        if ratio >= 0.25:
            return 0.40

        return min(
            ratio,
            0.25,
        )

    # =================================================================
    # Utilities
    # =================================================================

    @staticmethod
    def _safe_ratio(
        numerator: float,
        denominator: float,
    ) -> float:

        if denominator <= 0:
            return 0.0

        return max(
            numerator,
            0.0,
        ) / denominator

    @staticmethod
    def _money(
        value: Optional[float],
    ) -> float:

        if value is None:
            return 0.0

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
            value,
            0.0,
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


def create_default_financial_risk_model(
) -> FinancialRiskModel:

    return FinancialRiskModel(
        expected_loss_weight=0.35,
        asset_exposure_weight=0.25,
        operational_loss_weight=0.20,
        budget_impact_weight=0.10,
        predictive_risk_weight=0.10,
        default_reference_value=100_000.0,
    )


# =====================================================================
# Convenience API
# =====================================================================


def calculate_financial_risk(
    *,
    asset_id: str,
    failure_probability: float = 0.0,
    degradation_score: float = 0.0,
    rul_hours: Optional[float] = None,
    repair_cost: float = 0.0,
    replacement_cost: float = 0.0,
    downtime_cost: float = 0.0,
    production_loss: float = 0.0,
    emergency_cost: float = 0.0,
    asset_value: float = 0.0,
    recoverable_amount: float = 0.0,
) -> Dict[str, Any]:

    model = (
        create_default_financial_risk_model()
    )

    return model.predict_dict(
        asset_id=asset_id,

        failure_probability=(
            failure_probability
        ),

        degradation_score=(
            degradation_score
        ),

        rul_hours=rul_hours,

        repair_cost=repair_cost,
        replacement_cost=replacement_cost,

        downtime_cost=downtime_cost,
        production_loss=production_loss,

        emergency_cost=emergency_cost,

        asset_value=asset_value,

        recoverable_amount=(
            recoverable_amount
        ),
    )


# =====================================================================
# Demo
# =====================================================================


if __name__ == "__main__":

    from pprint import pprint

    model = (
        create_default_financial_risk_model()
    )

    result = model.predict_dict(
        asset_id="COMPRESSOR-001",

        failure_probability=0.82,
        degradation_score=0.76,
        rul_hours=140,

        repair_cost=18_000,
        replacement_cost=42_000,

        spare_parts_cost=4_500,
        labor_cost=3_000,

        downtime_cost=15_000,
        production_loss=28_000,

        energy_loss_cost=2_500,
        scrap_cost=5_000,

        emergency_cost=12_000,

        asset_value=75_000,

        recoverable_amount=5_000,

        maintenance_budget=25_000,

        asset_criticality=0.90,
        business_criticality=0.95,

        previous_risk_score=0.52,
    )

    pprint(result)