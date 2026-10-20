"""
app/ml/multimodal/anomaly_fusion.py

Multimodal anomaly fusion for industrial predictive-maintenance systems.

Purpose
-------
Combine heterogeneous anomaly signals from:

- sensor/telemetry models
- time-series models
- vibration analysis
- computer vision
- thermal analysis
- acoustic analysis
- maintenance history
- operating context
- rule-based detectors

into a normalized anomaly assessment.

Important safety rule
---------------------
This module DOES NOT perform emergency authorization or machine shutdown.

It produces evidence and risk signals.

Safety-critical actions must remain in:

    app/safety/
    app/decision_engine/
    app/services/

Architecture
------------
                 Telemetry
                    |
                 TS Model
                    |
 Vision --------> anomaly_fusion <-------- Vibration
                    |
 Thermal -----------+
                    |
 Maintenance -------+
                    |
              Fusion Assessment
                    |
             Decision Engine
             /             \
       maintenance       safety
        workflow         policy
"""


from __future__ import annotations

import math
import statistics

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


# ============================================================================
# Helpers
# ============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def finite_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


# ============================================================================
# Enums
# ============================================================================


class AnomalyModality(str, Enum):
    TELEMETRY = "telemetry"
    TIME_SERIES = "time_series"
    VIBRATION = "vibration"
    THERMAL = "thermal"
    COMPUTER_VISION = "computer_vision"
    ACOUSTIC = "acoustic"
    MAINTENANCE = "maintenance"
    ENVIRONMENTAL = "environmental"
    RULE_ENGINE = "rule_engine"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class AnomalyType(str, Enum):
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TREND = "trend"
    SENSOR_DRIFT = "sensor_drift"
    EQUIPMENT_DEGRADATION = "equipment_degradation"
    VISUAL_DEFECT = "visual_defect"
    THERMAL_ANOMALY = "thermal_anomaly"
    VIBRATION_ANOMALY = "vibration_anomaly"
    UNKNOWN = "unknown"


class AnomalySeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FusionMethod(str, Enum):
    WEIGHTED_SCORE = "weighted_score"
    MAX_EVIDENCE = "max_evidence"
    CONSENSUS = "consensus"
    BAYESIAN_LIKE = "bayesian_like"
    RULE_OVERRIDE = "rule_override"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============================================================================
# Raw anomaly signal
# ============================================================================


@dataclass(slots=True)
class AnomalySignal:
    """
    Normalized output from one anomaly detector.

    `score`
        Anomaly magnitude, 0..1.

    `confidence`
        Detector confidence, 0..1.

    `severity`
        Optional detector-provided severity.

    `weight`
        Reliability weight assigned to this modality/model.

    `features`
        Explainability information such as:
            temperature_zscore
            vibration_rms
            image_damage_score
            reconstruction_error
            forecast_residual
    """

    signal_id: str
    modality: AnomalyModality

    score: float
    confidence: float

    timestamp: datetime = field(
        default_factory=utc_now
    )

    anomaly_type: AnomalyType = AnomalyType.UNKNOWN

    severity: AnomalySeverity = (
        AnomalySeverity.NORMAL
    )

    weight: float = 1.0

    asset_id: Optional[str] = None
    sensor_id: Optional[str] = None
    camera_id: Optional[str] = None
    site_id: Optional[str] = None
    zone_id: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    features: dict[str, Any] = field(
        default_factory=dict
    )

    evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.score = clamp(
            finite_float(self.score)
        )

        self.confidence = clamp(
            finite_float(self.confidence)
        )

        self.weight = max(
            0.0,
            finite_float(self.weight, 1.0),
        )

    @property
    def effective_score(self) -> float:
        """
        Detector contribution after confidence adjustment.
        """

        return clamp(
            self.score * self.confidence
        )


# ============================================================================
# Fusion configuration
# ============================================================================


@dataclass(slots=True)
class ModalityWeight:
    modality: AnomalyModality
    weight: float

    min_confidence: float = 0.50
    enabled: bool = True


@dataclass(slots=True)
class FusionPolicy:
    """
    Configuration for multimodal fusion.

    These thresholds are decision-support thresholds, not emergency
    safety thresholds.
    """

    low_threshold: float = 0.25
    medium_threshold: float = 0.45
    high_threshold: float = 0.70
    critical_threshold: float = 0.90

    minimum_signal_confidence: float = 0.50

    consensus_min_modalities: int = 2
    consensus_threshold: float = 0.55

    strong_signal_threshold: float = 0.85

    # Agreement bonus when independent modalities agree.
    consensus_bonus: float = 0.10

    # Penalize weakly supported high anomaly scores.
    low_confidence_penalty: float = 0.10

    # Maximum number of signals retained in evidence.
    max_evidence_signals: int = 20

    modality_weights: tuple[ModalityWeight, ...] = (
        ModalityWeight(
            AnomalyModality.TELEMETRY,
            1.0,
        ),
        ModalityWeight(
            AnomalyModality.TIME_SERIES,
            1.1,
        ),
        ModalityWeight(
            AnomalyModality.VIBRATION,
            1.15,
        ),
        ModalityWeight(
            AnomalyModality.THERMAL,
            1.20,
        ),
        ModalityWeight(
            AnomalyModality.COMPUTER_VISION,
            1.0,
        ),
        ModalityWeight(
            AnomalyModality.ACOUSTIC,
            1.0,
        ),
        ModalityWeight(
            AnomalyModality.MAINTENANCE,
            0.75,
        ),
        ModalityWeight(
            AnomalyModality.ENVIRONMENTAL,
            0.80,
        ),
        ModalityWeight(
            AnomalyModality.RULE_ENGINE,
            1.25,
        ),
    )

    def weight_for(
        self,
        modality: AnomalyModality,
    ) -> float:
        for item in self.modality_weights:
            if (
                item.modality == modality
                and item.enabled
            ):
                return item.weight

        return 1.0


# ============================================================================
# Correlation
# ============================================================================


@dataclass(slots=True)
class FusionCorrelation:
    """
    Describes whether anomaly signals agree.

    Agreement is useful for confidence, but must never override
    deterministic safety rules.
    """

    modality_count: int

    modalities: list[AnomalyModality]

    mean_score: float
    max_score: float

    score_stddev: float

    consensus: bool
    independent_evidence: bool

    agreement_score: float


# ============================================================================
# Fusion assessment
# ============================================================================


@dataclass(slots=True)
class AnomalyFusionAssessment:
    assessment_id: str

    timestamp: datetime

    asset_id: Optional[str]

    fused_score: float
    confidence: float

    severity: AnomalySeverity
    confidence_level: ConfidenceLevel

    anomaly_types: list[AnomalyType]

    modalities: list[AnomalyModality]

    primary_modality: AnomalyModality

    fusion_method: FusionMethod

    correlation: FusionCorrelation

    signals: list[AnomalySignal]

    evidence: list[str]

    contributing_features: dict[str, Any]

    requires_investigation: bool
    requires_maintenance_review: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_anomalous(self) -> bool:
        return (
            self.severity
            != AnomalySeverity.NORMAL
        )

    @property
    def is_high_risk(self) -> bool:
        return self.severity in {
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL,
        }


# ============================================================================
# Fusion engine
# ============================================================================


class AnomalyFusionEngine:
    """
    Multimodal anomaly fusion engine.

    The engine is intentionally deterministic.

    It should sit downstream of individual ML detectors and upstream
    of the central decision/safety layer.
    """

    def __init__(
        self,
        policy: Optional[FusionPolicy] = None,
    ) -> None:
        self.policy = (
            policy
            or FusionPolicy()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        signals: Iterable[AnomalySignal],
        *,
        asset_id: Optional[str] = None,
    ) -> AnomalyFusionAssessment:
        normalized = self._prepare_signals(
            signals,
            asset_id=asset_id,
        )

        if not normalized:
            return self._empty_assessment(
                asset_id=asset_id
            )

        correlation = self._correlate(
            normalized
        )

        fused_score = self._fuse_score(
            normalized,
            correlation,
        )

        confidence = self._fuse_confidence(
            normalized,
            correlation,
        )

        severity = self._severity(
            fused_score,
            normalized,
        )

        anomaly_types = self._anomaly_types(
            normalized
        )

        modalities = list(
            dict.fromkeys(
                signal.modality
                for signal in normalized
            )
        )

        primary_modality = self._primary_modality(
            normalized
        )

        evidence = self._evidence(
            normalized
        )

        features = self._features(
            normalized
        )

        return AnomalyFusionAssessment(
            assessment_id=self._new_id(),
            timestamp=utc_now(),
            asset_id=asset_id
            or self._infer_asset_id(normalized),

            fused_score=fused_score,
            confidence=confidence,

            severity=severity,
            confidence_level=self._confidence_level(
                confidence
            ),

            anomaly_types=anomaly_types,
            modalities=modalities,
            primary_modality=primary_modality,

            fusion_method=(
                FusionMethod.WEIGHTED_SCORE
            ),

            correlation=correlation,

            signals=normalized,
            evidence=evidence,
            contributing_features=features,

            requires_investigation=(
                severity
                in {
                    AnomalySeverity.MEDIUM,
                    AnomalySeverity.HIGH,
                    AnomalySeverity.CRITICAL,
                }
            ),

            requires_maintenance_review=(
                severity
                in {
                    AnomalySeverity.HIGH,
                    AnomalySeverity.CRITICAL,
                }
                or any(
                    signal.anomaly_type
                    == AnomalyType.EQUIPMENT_DEGRADATION
                    for signal in normalized
                )
            ),

            metadata={
                "signal_count": len(normalized),
                "modality_count": len(modalities),
            },
        )

    def fuse_batch(
        self,
        signal_groups: Mapping[
            str,
            Iterable[AnomalySignal]
        ],
    ) -> dict[str, AnomalyFusionAssessment]:
        return {
            asset_id: self.fuse(
                signals,
                asset_id=asset_id,
            )
            for asset_id, signals
            in signal_groups.items()
        }

    def compare(
        self,
        current: AnomalyFusionAssessment,
        previous: Optional[AnomalyFusionAssessment],
    ) -> dict[str, Any]:
        """
        Compare two fused assessments.

        Useful for trend detection and dashboard explanations.
        """

        if previous is None:
            return {
                "score_change": current.fused_score,
                "confidence_change": current.confidence,
                "severity_changed": True,
                "direction": "new",
            }

        score_change = (
            current.fused_score
            - previous.fused_score
        )

        confidence_change = (
            current.confidence
            - previous.confidence
        )

        if score_change > 0.05:
            direction = "worsening"
        elif score_change < -0.05:
            direction = "improving"
        else:
            direction = "stable"

        return {
            "score_change": score_change,
            "confidence_change": confidence_change,
            "severity_changed": (
                current.severity
                != previous.severity
            ),
            "direction": direction,
            "previous_severity": (
                previous.severity.value
            ),
            "current_severity": (
                current.severity.value
            ),
        }

    # ------------------------------------------------------------------
    # Signal preparation
    # ------------------------------------------------------------------

    def _prepare_signals(
        self,
        signals: Iterable[AnomalySignal],
        *,
        asset_id: Optional[str],
    ) -> list[AnomalySignal]:
        result: list[AnomalySignal] = []

        for signal in signals:
            if (
                signal.confidence
                < self.policy.minimum_signal_confidence
            ):
                continue

            if (
                asset_id
                and signal.asset_id
                and signal.asset_id != asset_id
            ):
                continue

            signal_weight = (
                signal.weight
                * self.policy.weight_for(
                    signal.modality
                )
            )

            if signal_weight <= 0:
                continue

            signal.weight = signal_weight

            result.append(signal)

        return result

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def _correlate(
        self,
        signals: list[AnomalySignal],
    ) -> FusionCorrelation:
        modalities = list(
            dict.fromkeys(
                signal.modality
                for signal in signals
            )
        )

        scores = [
            signal.score
            for signal in signals
        ]

        mean_score = (
            statistics.fmean(scores)
            if scores
            else 0.0
        )

        max_score = (
            max(scores)
            if scores
            else 0.0
        )

        stddev = (
            statistics.pstdev(scores)
            if len(scores) > 1
            else 0.0
        )

        independent_evidence = (
            len(modalities) >= 2
        )

        consensus = (
            len(modalities)
            >= self.policy.consensus_min_modalities
            and mean_score
            >= self.policy.consensus_threshold
        )

        # Agreement score:
        # high when multiple modalities have similar anomaly scores.
        agreement_score = 1.0

        if len(scores) > 1:
            agreement_score = clamp(
                1.0 - stddev
            )

        return FusionCorrelation(
            modality_count=len(modalities),
            modalities=modalities,
            mean_score=mean_score,
            max_score=max_score,
            score_stddev=stddev,
            consensus=consensus,
            independent_evidence=independent_evidence,
            agreement_score=agreement_score,
        )

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    def _fuse_score(
        self,
        signals: list[AnomalySignal],
        correlation: FusionCorrelation,
    ) -> float:
        if not signals:
            return 0.0

        numerator = sum(
            signal.effective_score
            * signal.weight
            for signal in signals
        )

        denominator = sum(
            signal.weight
            for signal in signals
        )

        if denominator <= 0:
            return 0.0

        weighted_score = (
            numerator / denominator
        )

        # Strongest evidence protects against dilution.
        max_evidence = correlation.max_score

        # Weighted score captures consensus.
        # Max evidence prevents one serious detector from being
        # completely hidden by several normal detectors.
        fused = (
            0.70 * weighted_score
            + 0.30 * max_evidence
        )

        if correlation.consensus:
            fused += (
                self.policy.consensus_bonus
                * correlation.agreement_score
            )

        # Prevent score inflation beyond [0, 1].
        return clamp(fused)

    # ------------------------------------------------------------------
    # Confidence fusion
    # ------------------------------------------------------------------

    def _fuse_confidence(
        self,
        signals: list[AnomalySignal],
        correlation: FusionCorrelation,
    ) -> float:
        if not signals:
            return 0.0

        weights = [
            max(signal.weight, 0.0001)
            for signal in signals
        ]

        weighted_confidence = (
            sum(
                signal.confidence * weight
                for signal, weight
                in zip(signals, weights)
            )
            / sum(weights)
        )

        confidence = weighted_confidence

        if correlation.independent_evidence:
            confidence += 0.05

        if correlation.consensus:
            confidence += 0.10 * (
                correlation.agreement_score
            )

        if (
            len(signals) == 1
            and signals[0].confidence < 0.70
        ):
            confidence -= (
                self.policy.low_confidence_penalty
            )

        return clamp(confidence)

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    def _severity(
        self,
        score: float,
        signals: list[AnomalySignal],
    ) -> AnomalySeverity:
        """
        Convert fused anomaly score into anomaly severity.

        This is not the final safety severity.

        A thermal/fire safety event must still pass through the
        deterministic safety policy.
        """

        if any(
            signal.severity
            == AnomalySeverity.CRITICAL
            for signal in signals
        ):
            return AnomalySeverity.CRITICAL

        if score >= self.policy.critical_threshold:
            return AnomalySeverity.CRITICAL

        if any(
            signal.severity
            == AnomalySeverity.HIGH
            for signal in signals
        ) and score >= self.policy.high_threshold:
            return AnomalySeverity.HIGH

        if score >= self.policy.high_threshold:
            return AnomalySeverity.HIGH

        if score >= self.policy.medium_threshold:
            return AnomalySeverity.MEDIUM

        if score >= self.policy.low_threshold:
            return AnomalySeverity.LOW

        return AnomalySeverity.NORMAL

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def _anomaly_types(
        self,
        signals: list[AnomalySignal],
    ) -> list[AnomalyType]:
        types = list(
            dict.fromkeys(
                signal.anomaly_type
                for signal in signals
                if (
                    signal.anomaly_type
                    != AnomalyType.UNKNOWN
                )
            )
        )

        return types or [
            AnomalyType.UNKNOWN
        ]

    def _primary_modality(
        self,
        signals: list[AnomalySignal],
    ) -> AnomalyModality:
        if not signals:
            return AnomalyModality.UNKNOWN

        return max(
            signals,
            key=lambda signal: (
                signal.effective_score
                * signal.weight
            ),
        ).modality

    def _evidence(
        self,
        signals: list[AnomalySignal],
    ) -> list[str]:
        evidence: list[str] = []

        ordered = sorted(
            signals,
            key=lambda signal: (
                signal.effective_score
                * signal.weight
            ),
            reverse=True,
        )

        for signal in ordered[
            : self.policy.max_evidence_signals
        ]:
            prefix = (
                f"{signal.modality.value}: "
                f"score={signal.score:.3f}, "
                f"confidence={signal.confidence:.3f}"
            )

            if signal.model_name:
                prefix += (
                    f", model={signal.model_name}"
                )

            evidence.append(prefix)

            evidence.extend(
                signal.evidence[:3]
            )

        return evidence

    def _features(
        self,
        signals: list[AnomalySignal],
    ) -> dict[str, Any]:
        features: dict[str, Any] = {}

        for signal in signals:
            prefix = signal.modality.value

            for key, value in signal.features.items():
                features[
                    f"{prefix}.{key}"
                ] = value

        return features

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _confidence_level(
        self,
        confidence: float,
    ) -> ConfidenceLevel:
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH

        if confidence >= 0.65:
            return ConfidenceLevel.MEDIUM

        return ConfidenceLevel.LOW

    def _infer_asset_id(
        self,
        signals: list[AnomalySignal],
    ) -> Optional[str]:
        asset_ids = [
            signal.asset_id
            for signal in signals
            if signal.asset_id
        ]

        if not asset_ids:
            return None

        return asset_ids[0]

    def _empty_assessment(
        self,
        *,
        asset_id: Optional[str],
    ) -> AnomalyFusionAssessment:
        correlation = FusionCorrelation(
            modality_count=0,
            modalities=[],
            mean_score=0.0,
            max_score=0.0,
            score_stddev=0.0,
            consensus=False,
            independent_evidence=False,
            agreement_score=0.0,
        )

        return AnomalyFusionAssessment(
            assessment_id=self._new_id(),
            timestamp=utc_now(),
            asset_id=asset_id,
            fused_score=0.0,
            confidence=0.0,
            severity=AnomalySeverity.NORMAL,
            confidence_level=ConfidenceLevel.LOW,
            anomaly_types=[
                AnomalyType.UNKNOWN
            ],
            modalities=[],
            primary_modality=AnomalyModality.UNKNOWN,
            fusion_method=FusionMethod.WEIGHTED_SCORE,
            correlation=correlation,
            signals=[],
            evidence=[],
            contributing_features={},
            requires_investigation=False,
            requires_maintenance_review=False,
            metadata={
                "signal_count": 0,
                "modality_count": 0,
            },
        )

    @staticmethod
    def _new_id() -> str:
        import uuid

        return str(uuid.uuid4())


# ============================================================================
# Convenience constructors
# ============================================================================


def telemetry_signal(
    *,
    signal_id: str,
    score: float,
    confidence: float,
    asset_id: Optional[str] = None,
    sensor_id: Optional[str] = None,
    anomaly_type: AnomalyType = (
        AnomalyType.POINT
    ),
    features: Optional[
        Mapping[str, Any]
    ] = None,
    evidence: Optional[
        Iterable[str]
    ] = None,
) -> AnomalySignal:
    return AnomalySignal(
        signal_id=signal_id,
        modality=AnomalyModality.TELEMETRY,
        score=score,
        confidence=confidence,
        asset_id=asset_id,
        sensor_id=sensor_id,
        anomaly_type=anomaly_type,
        features=dict(features or {}),
        evidence=list(evidence or []),
    )


def time_series_signal(
    *,
    signal_id: str,
    score: float,
    confidence: float,
    asset_id: Optional[str] = None,
    sensor_id: Optional[str] = None,
    anomaly_type: AnomalyType = (
        AnomalyType.TREND
    ),
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    features: Optional[
        Mapping[str, Any]
    ] = None,
) -> AnomalySignal:
    return AnomalySignal(
        signal_id=signal_id,
        modality=AnomalyModality.TIME_SERIES,
        score=score,
        confidence=confidence,
        asset_id=asset_id,
        sensor_id=sensor_id,
        anomaly_type=anomaly_type,
        model_name=model_name,
        model_version=model_version,
        features=dict(features or {}),
    )


def vision_signal(
    *,
    signal_id: str,
    score: float,
    confidence: float,
    asset_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    anomaly_type: AnomalyType = (
        AnomalyType.VISUAL_DEFECT
    ),
    features: Optional[
        Mapping[str, Any]
    ] = None,
    evidence: Optional[
        Iterable[str]
    ] = None,
) -> AnomalySignal:
    return AnomalySignal(
        signal_id=signal_id,
        modality=AnomalyModality.COMPUTER_VISION,
        score=score,
        confidence=confidence,
        asset_id=asset_id,
        camera_id=camera_id,
        anomaly_type=anomaly_type,
        features=dict(features or {}),
        evidence=list(evidence or []),
    )


def vibration_signal(
    *,
    signal_id: str,
    score: float,
    confidence: float,
    asset_id: Optional[str] = None,
    sensor_id: Optional[str] = None,
    features: Optional[
        Mapping[str, Any]
    ] = None,
) -> AnomalySignal:
    return AnomalySignal(
        signal_id=signal_id,
        modality=AnomalyModality.VIBRATION,
        score=score,
        confidence=confidence,
        asset_id=asset_id,
        sensor_id=sensor_id,
        anomaly_type=(
            AnomalyType.VIBRATION_ANOMALY
        ),
        features=dict(features or {}),
    )


def thermal_signal(
    *,
    signal_id: str,
    score: float,
    confidence: float,
    asset_id: Optional[str] = None,
    sensor_id: Optional[str] = None,
    features: Optional[
        Mapping[str, Any]
    ] = None,
) -> AnomalySignal:
    return AnomalySignal(
        signal_id=signal_id,
        modality=AnomalyModality.THERMAL,
        score=score,
        confidence=confidence,
        asset_id=asset_id,
        sensor_id=sensor_id,
        anomaly_type=(
            AnomalyType.THERMAL_ANOMALY
        ),
        features=dict(features or {}),
    )


# ============================================================================
# Default engine
# ============================================================================


default_anomaly_fusion_engine = (
    AnomalyFusionEngine()
)