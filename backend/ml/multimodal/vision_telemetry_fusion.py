"""
Vision + Telemetry Fusion Engine
================================

Combines visual AI detections with industrial sensor telemetry to produce
a unified asset/safety risk assessment.

Inputs
------
Vision:
    - fire
    - smoke
    - overheating
    - crack
    - corrosion
    - leakage
    - defect
    - person / restricted-zone events

Telemetry:
    - temperature
    - vibration
    - pressure
    - current
    - voltage
    - energy / power

Outputs
-------
    - vision risk
    - telemetry risk
    - fused risk
    - severity
    - confidence
    - contributing signals
    - recommended actions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import exp
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class RiskSeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class VisionSignal:
    label: str
    confidence: float
    severity: float = 1.0
    source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetrySignal:
    name: str
    value: float
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    minimum_warning: Optional[float] = None
    minimum_critical: Optional[float] = None
    anomaly_score: Optional[float] = None
    unit: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    timestamp: str
    vision_risk: float
    telemetry_risk: float
    fused_risk: float
    confidence: float
    severity: RiskSeverity
    vision_events: List[Dict[str, Any]]
    telemetry_events: List[Dict[str, Any]]
    contributing_signals: List[str]
    recommended_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "vision_risk": round(self.vision_risk, 4),
            "telemetry_risk": round(self.telemetry_risk, 4),
            "fused_risk": round(self.fused_risk, 4),
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "vision_events": self.vision_events,
            "telemetry_events": self.telemetry_events,
            "contributing_signals": self.contributing_signals,
            "recommended_actions": self.recommended_actions,
        }


class VisionTelemetryFusion:
    """
    Deterministic multimodal risk fusion engine.

    The engine deliberately does not trigger physical emergency controls.
    It produces a risk decision that higher-level safety/emergency services
    can validate and act upon.
    """

    DEFAULT_VISION_WEIGHTS: Dict[str, float] = {
        "fire": 1.00,
        "smoke": 0.90,
        "overheating": 0.90,
        "leakage": 0.85,
        "crack": 0.75,
        "corrosion": 0.60,
        "defect": 0.60,
        "restricted_zone": 0.80,
        "unsafe_proximity": 0.85,
        "person": 0.35,
    }

    CRITICAL_LABELS = {
        "fire",
        "smoke",
        "overheating",
        "unsafe_proximity",
    }

    def __init__(
        self,
        vision_weight: float = 0.50,
        telemetry_weight: float = 0.50,
        min_vision_confidence: float = 0.30,
        vision_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        if vision_weight < 0 or telemetry_weight < 0:
            raise ValueError("Fusion weights cannot be negative.")

        total = vision_weight + telemetry_weight
        if total <= 0:
            raise ValueError("At least one fusion weight must be positive.")

        self.vision_weight = vision_weight / total
        self.telemetry_weight = telemetry_weight / total
        self.min_vision_confidence = self._clamp(min_vision_confidence)

        self.vision_weights = dict(self.DEFAULT_VISION_WEIGHTS)

        if vision_weights:
            for label, weight in vision_weights.items():
                self.vision_weights[label.lower()] = self._clamp(weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        vision_signals: Sequence[VisionSignal],
        telemetry_signals: Sequence[TelemetrySignal],
    ) -> FusionResult:

        vision_risk, vision_events = self._calculate_vision_risk(
            vision_signals
        )

        telemetry_risk, telemetry_events = (
            self._calculate_telemetry_risk(telemetry_signals)
        )

        fused_risk = (
            self.vision_weight * vision_risk
            + self.telemetry_weight * telemetry_risk
        )

        # Cross-modal agreement increases confidence/risk.
        correlation_bonus = self._cross_modal_bonus(
            vision_events,
            telemetry_events,
        )

        fused_risk = self._clamp(fused_risk + correlation_bonus)

        # Safety rule:
        # a highly confident critical visual signal should not disappear
        # because normal telemetry diluted the weighted average.
        critical_visual = max(
            (
                event["risk"]
                for event in vision_events
                if event["label"] in self.CRITICAL_LABELS
            ),
            default=0.0,
        )

        if critical_visual >= 0.85:
            fused_risk = max(fused_risk, critical_visual * 0.90)

        confidence = self._fusion_confidence(
            vision_events,
            telemetry_events,
            vision_risk,
            telemetry_risk,
        )

        severity = self._severity(fused_risk)

        contributors = self._contributors(
            vision_events,
            telemetry_events,
        )

        actions = self._recommended_actions(
            severity,
            vision_events,
            telemetry_events,
        )

        return FusionResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            vision_risk=vision_risk,
            telemetry_risk=telemetry_risk,
            fused_risk=fused_risk,
            confidence=confidence,
            severity=severity,
            vision_events=vision_events,
            telemetry_events=telemetry_events,
            contributing_signals=contributors,
            recommended_actions=actions,
        )

    def fuse_dict(
        self,
        vision: Iterable[Mapping[str, Any]],
        telemetry: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:

        vision_signals = [
            VisionSignal(
                label=str(item["label"]),
                confidence=float(item.get("confidence", 0.0)),
                severity=float(item.get("severity", 1.0)),
                source=item.get("source"),
                metadata=item.get("metadata", {}),
            )
            for item in vision
        ]

        telemetry_signals = [
            TelemetrySignal(
                name=str(item["name"]),
                value=float(item["value"]),
                warning_threshold=self._optional_float(
                    item.get("warning_threshold")
                ),
                critical_threshold=self._optional_float(
                    item.get("critical_threshold")
                ),
                minimum_warning=self._optional_float(
                    item.get("minimum_warning")
                ),
                minimum_critical=self._optional_float(
                    item.get("minimum_critical")
                ),
                anomaly_score=self._optional_float(
                    item.get("anomaly_score")
                ),
                unit=item.get("unit"),
                metadata=item.get("metadata", {}),
            )
            for item in telemetry
        ]

        return self.fuse(
            vision_signals,
            telemetry_signals,
        ).to_dict()

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def _calculate_vision_risk(
        self,
        signals: Sequence[VisionSignal],
    ) -> tuple[float, List[Dict[str, Any]]]:

        events: List[Dict[str, Any]] = []

        for signal in signals:
            confidence = self._clamp(signal.confidence)

            if confidence < self.min_vision_confidence:
                continue

            label = signal.label.lower().strip()

            label_weight = self.vision_weights.get(label, 0.50)
            severity = self._clamp(signal.severity)

            risk = self._clamp(
                confidence * label_weight * severity
            )

            events.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "risk": risk,
                    "source": signal.source,
                    "metadata": dict(signal.metadata),
                }
            )

        if not events:
            return 0.0, []

        # Noisy-OR aggregation:
        # several moderate detections increase total risk without simple
        # unbounded addition.
        risk = self._noisy_or(
            event["risk"] for event in events
        )

        return risk, events

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _calculate_telemetry_risk(
        self,
        signals: Sequence[TelemetrySignal],
    ) -> tuple[float, List[Dict[str, Any]]]:

        events: List[Dict[str, Any]] = []

        for signal in signals:

            threshold_risk = self._threshold_risk(signal)

            anomaly_risk = (
                self._clamp(signal.anomaly_score)
                if signal.anomaly_score is not None
                else 0.0
            )

            risk = max(threshold_risk, anomaly_risk)

            if risk <= 0:
                continue

            events.append(
                {
                    "name": signal.name,
                    "value": signal.value,
                    "unit": signal.unit,
                    "threshold_risk": threshold_risk,
                    "anomaly_score": anomaly_risk,
                    "risk": risk,
                    "metadata": dict(signal.metadata),
                }
            )

        if not events:
            return 0.0, []

        risk = self._noisy_or(
            event["risk"] for event in events
        )

        return risk, events

    def _threshold_risk(
        self,
        signal: TelemetrySignal,
    ) -> float:

        value = signal.value

        # Upper-bound thresholds.
        if (
            signal.critical_threshold is not None
            and value >= signal.critical_threshold
        ):
            return 1.0

        if (
            signal.warning_threshold is not None
            and value >= signal.warning_threshold
        ):
            if signal.critical_threshold is None:
                return 0.65

            denominator = (
                signal.critical_threshold
                - signal.warning_threshold
            )

            if denominator <= 0:
                return 0.75

            progress = (
                value - signal.warning_threshold
            ) / denominator

            return self._clamp(
                0.60 + 0.40 * progress
            )

        # Lower-bound thresholds.
        if (
            signal.minimum_critical is not None
            and value <= signal.minimum_critical
        ):
            return 1.0

        if (
            signal.minimum_warning is not None
            and value <= signal.minimum_warning
        ):
            if signal.minimum_critical is None:
                return 0.65

            denominator = (
                signal.minimum_warning
                - signal.minimum_critical
            )

            if denominator <= 0:
                return 0.75

            progress = (
                signal.minimum_warning - value
            ) / denominator

            return self._clamp(
                0.60 + 0.40 * progress
            )

        return 0.0

    # ------------------------------------------------------------------
    # Cross-modal reasoning
    # ------------------------------------------------------------------

    def _cross_modal_bonus(
        self,
        vision_events: Sequence[Mapping[str, Any]],
        telemetry_events: Sequence[Mapping[str, Any]],
    ) -> float:

        vision_labels = {
            str(event["label"])
            for event in vision_events
        }

        telemetry_names = {
            str(event["name"]).lower()
            for event in telemetry_events
        }

        bonus = 0.0

        # Fire/smoke + thermal evidence.
        if (
            {"fire", "smoke"} & vision_labels
            and self._contains_any(
                telemetry_names,
                {"temperature", "thermal", "heat"},
            )
        ):
            bonus += 0.15

        # Visual overheating + high temperature/current/power.
        if (
            "overheating" in vision_labels
            and self._contains_any(
                telemetry_names,
                {
                    "temperature",
                    "current",
                    "power",
                    "energy",
                },
            )
        ):
            bonus += 0.12

        # Crack/defect + abnormal vibration.
        if (
            {"crack", "defect"} & vision_labels
            and self._contains_any(
                telemetry_names,
                {"vibration"},
            )
        ):
            bonus += 0.10

        # Leakage + pressure anomaly.
        if (
            "leakage" in vision_labels
            and self._contains_any(
                telemetry_names,
                {"pressure"},
            )
        ):
            bonus += 0.12

        return min(bonus, 0.25)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _fusion_confidence(
        self,
        vision_events: Sequence[Mapping[str, Any]],
        telemetry_events: Sequence[Mapping[str, Any]],
        vision_risk: float,
        telemetry_risk: float,
    ) -> float:

        vision_confidence = max(
            (
                float(event["confidence"])
                for event in vision_events
            ),
            default=0.0,
        )

        telemetry_confidence = max(
            (
                float(event["risk"])
                for event in telemetry_events
            ),
            default=0.0,
        )

        if vision_events and telemetry_events:
            base = (
                vision_confidence + telemetry_confidence
            ) / 2

            agreement = 1.0 - abs(
                vision_risk - telemetry_risk
            )

            return self._clamp(
                0.75 * base + 0.25 * agreement
            )

        return self._clamp(
            max(vision_confidence, telemetry_confidence)
        )

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    @staticmethod
    def _severity(risk: float) -> RiskSeverity:

        if risk >= 0.85:
            return RiskSeverity.CRITICAL

        if risk >= 0.70:
            return RiskSeverity.HIGH

        if risk >= 0.45:
            return RiskSeverity.MEDIUM

        if risk >= 0.20:
            return RiskSeverity.LOW

        return RiskSeverity.NORMAL

    def _contributors(
        self,
        vision_events: Sequence[Mapping[str, Any]],
        telemetry_events: Sequence[Mapping[str, Any]],
    ) -> List[str]:

        contributors: List[tuple[str, float]] = []

        for event in vision_events:
            contributors.append(
                (
                    f"vision:{event['label']}",
                    float(event["risk"]),
                )
            )

        for event in telemetry_events:
            contributors.append(
                (
                    f"telemetry:{event['name']}",
                    float(event["risk"]),
                )
            )

        contributors.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            name
            for name, risk in contributors
            if risk >= 0.20
        ]

    def _recommended_actions(
        self,
        severity: RiskSeverity,
        vision_events: Sequence[Mapping[str, Any]],
        telemetry_events: Sequence[Mapping[str, Any]],
    ) -> List[str]:

        labels = {
            str(event["label"])
            for event in vision_events
        }

        actions: List[str] = []

        if "fire" in labels:
            actions.extend(
                [
                    "Trigger fire incident verification workflow.",
                    "Notify the emergency response service.",
                    "Escalate for immediate human confirmation.",
                ]
            )

        if "smoke" in labels:
            actions.append(
                "Inspect the affected zone for smoke or combustion."
            )

        if "overheating" in labels:
            actions.append(
                "Inspect equipment thermal condition and load."
            )

        if "leakage" in labels:
            actions.append(
                "Inspect fluid/gas system and pressure telemetry."
            )

        if {"crack", "defect"} & labels:
            actions.append(
                "Schedule priority visual and mechanical inspection."
            )

        if severity == RiskSeverity.CRITICAL:
            actions.append(
                "Escalate incident to the critical-response workflow."
            )

        elif severity == RiskSeverity.HIGH:
            actions.append(
                "Create a high-priority maintenance/safety incident."
            )

        elif severity == RiskSeverity.MEDIUM:
            actions.append(
                "Increase monitoring frequency and schedule inspection."
            )

        elif severity == RiskSeverity.LOW:
            actions.append(
                "Continue monitoring and record the anomaly."
            )

        if not actions:
            actions.append(
                "Continue normal monitoring."
            )

        return list(dict.fromkeys(actions))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _noisy_or(values: Iterable[float]) -> float:

        probability_none = 1.0

        for value in values:
            value = max(0.0, min(1.0, float(value)))
            probability_none *= 1.0 - value

        return 1.0 - probability_none

    @staticmethod
    def _contains_any(
        names: Iterable[str],
        keywords: set[str],
    ) -> bool:

        return any(
            keyword in name
            for name in names
            for keyword in keywords
        )

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        return float(value)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------

def fuse_vision_telemetry(
    vision: Iterable[Mapping[str, Any]],
    telemetry: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Simple application-facing interface.
    """

    engine = VisionTelemetryFusion()

    return engine.fuse_dict(
        vision=vision,
        telemetry=telemetry,
    )


# ----------------------------------------------------------------------
# Local demonstration
# ----------------------------------------------------------------------

if __name__ == "__main__":

    vision_data = [
        {
            "label": "overheating",
            "confidence": 0.94,
            "severity": 0.95,
            "source": "thermal-camera-01",
        },
        {
            "label": "smoke",
            "confidence": 0.81,
            "severity": 0.90,
            "source": "camera-04",
        },
    ]

    telemetry_data = [
        {
            "name": "bearing_temperature",
            "value": 96.0,
            "unit": "C",
            "warning_threshold": 75.0,
            "critical_threshold": 100.0,
            "anomaly_score": 0.88,
        },
        {
            "name": "motor_current",
            "value": 42.0,
            "unit": "A",
            "warning_threshold": 35.0,
            "critical_threshold": 50.0,
            "anomaly_score": 0.74,
        },
        {
            "name": "vibration_rms",
            "value": 9.2,
            "unit": "mm/s",
            "warning_threshold": 7.0,
            "critical_threshold": 11.0,
            "anomaly_score": 0.78,
        },
    ]

    result = fuse_vision_telemetry(
        vision_data,
        telemetry_data,
    )

    from pprint import pprint

    pprint(result)