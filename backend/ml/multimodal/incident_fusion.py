# app/ml/multimodal/incident_fusion.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import exp, isfinite
from statistics import mean
from typing import Any, Iterable, Sequence
from uuid import uuid4


# ============================================================
# Enums
# ============================================================


class IncidentType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    THERMAL_EVENT = "thermal_event"
    ELECTRICAL_INCIDENT = "electrical_incident"
    GAS_LEAK = "gas_leak"
    LIQUID_LEAK = "liquid_leak"
    PRESSURE_INCIDENT = "pressure_incident"

    MECHANICAL_FAILURE = "mechanical_failure"
    BEARING_FAILURE_SUSPECTED = "bearing_failure_suspected"
    GEAR_FAILURE_SUSPECTED = "gear_failure_suspected"
    STRUCTURAL_DAMAGE = "structural_damage"

    SAFETY_ZONE_VIOLATION = "safety_zone_violation"
    PERSON_PROXIMITY = "person_proximity"
    ABNORMAL_MOTION = "abnormal_motion"

    EQUIPMENT_DAMAGE = "equipment_damage"
    ENVIRONMENTAL_HAZARD = "environmental_hazard"

    MULTI_SIGNAL_HAZARD = "multi_signal_hazard"
    UNKNOWN = "unknown"


class IncidentSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentState(str, Enum):
    CANDIDATE = "candidate"
    SUSPECTED = "suspected"
    CORROBORATED = "corroborated"
    CONFLICTING = "conflicting"
    CRITICAL = "critical"


class IncidentSource(str, Enum):
    TELEMETRY = "telemetry"
    SENSOR = "sensor"
    ANOMALY_DETECTOR = "anomaly_detector"
    COMPUTER_VISION = "computer_vision"
    THERMAL_CAMERA = "thermal_camera"
    ACOUSTIC = "acoustic"
    IMAGE_SENSOR_FUSION = "image_sensor_fusion"
    ANOMALY_FUSION = "anomaly_fusion"
    SAFETY_ENGINE = "safety_engine"
    RULE_ENGINE = "rule_engine"
    OPERATOR = "operator"
    EXTERNAL_SYSTEM = "external_system"
    UNKNOWN = "unknown"


class EvidenceKind(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    TEMPERATURE = "temperature"
    ELECTRICAL = "electrical"
    GAS = "gas"
    LEAK = "leak"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    ACOUSTIC = "acoustic"
    BEARING = "bearing"
    GEAR = "gear"
    STRUCTURAL = "structural"
    PERSON = "person"
    PROXIMITY = "proximity"
    RESTRICTED_ZONE = "restricted_zone"
    MOTION = "motion"
    DAMAGE = "damage"
    ENVIRONMENTAL = "environmental"
    UNKNOWN = "unknown"


class FusionDecision(str, Enum):
    IGNORE = "ignore"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    MAINTENANCE_REVIEW = "maintenance_review"
    SAFETY_REVIEW = "safety_review"
    ESCALATION_REVIEW = "escalation_review"


# ============================================================
# Evidence
# ============================================================


@dataclass(slots=True)
class IncidentEvidence:
    evidence_id: str
    timestamp: datetime

    source: IncidentSource
    kind: EvidenceKind

    score: float
    confidence: float

    asset_id: str | None = None
    site_id: str | None = None
    zone_id: str | None = None
    component_id: str | None = None

    source_id: str | None = None

    severity: IncidentSeverity = IncidentSeverity.LOW

    description: str = ""

    independent_source_key: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = _clamp(self.score)
        self.confidence = _clamp(self.confidence)

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(
                tzinfo=timezone.utc
            )

    @property
    def effective_score(self) -> float:
        return _clamp(
            self.score * self.confidence
        )


# ============================================================
# Correlation
# ============================================================


@dataclass(slots=True)
class IncidentCorrelation:
    evidence_count: int

    source_count: int
    independent_source_count: int
    evidence_kind_count: int

    temporal_agreement: float
    spatial_agreement: float
    semantic_agreement: float
    confidence_agreement: float

    corroborated: bool
    conflicting: bool

    earliest_timestamp: datetime | None
    latest_timestamp: datetime | None

    duration_seconds: float

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Incident Hypothesis
# ============================================================


@dataclass(slots=True)
class IncidentHypothesis:
    hypothesis_id: str

    incident_type: IncidentType

    score: float
    confidence: float
    severity: IncidentSeverity

    supporting_evidence_ids: list[str]

    rationale: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Fused Incident
# ============================================================


@dataclass(slots=True)
class FusedIncident:
    incident_id: str

    created_at: datetime
    updated_at: datetime

    incident_type: IncidentType
    state: IncidentState
    severity: IncidentSeverity

    risk_score: float
    confidence: float

    asset_id: str | None
    site_id: str | None
    zone_id: str | None
    component_id: str | None

    evidence: list[IncidentEvidence]
    correlation: IncidentCorrelation

    hypotheses: list[IncidentHypothesis]

    primary_hypothesis: IncidentHypothesis | None

    decision: FusionDecision

    requires_investigation: bool
    requires_maintenance_review: bool
    requires_safety_review: bool
    requires_escalation_review: bool

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_corroborated(self) -> bool:
        return self.correlation.corroborated

    @property
    def is_critical(self) -> bool:
        return self.severity == IncidentSeverity.CRITICAL

    @property
    def is_high_risk(self) -> bool:
        return self.severity in {
            IncidentSeverity.HIGH,
            IncidentSeverity.CRITICAL,
        }


# ============================================================
# Policy
# ============================================================


@dataclass(slots=True)
class IncidentFusionPolicy:
    correlation_window_seconds: float = 60.0

    minimum_evidence_confidence: float = 0.50
    minimum_evidence_score: float = 0.20

    minimum_independent_sources: int = 2

    temporal_weight: float = 0.25
    spatial_weight: float = 0.25
    semantic_weight: float = 0.30
    confidence_weight: float = 0.20

    evidence_weight: float = 0.65
    correlation_weight: float = 0.25
    diversity_weight: float = 0.10

    corroboration_bonus: float = 0.10
    conflict_penalty: float = 0.15

    low_threshold: float = 0.25
    medium_threshold: float = 0.45
    high_threshold: float = 0.70
    critical_threshold: float = 0.90

    duplicate_time_seconds: float = 120.0

    max_evidence: int = 100


# ============================================================
# Incident Semantic Rules
# ============================================================


_INCIDENT_RULES: dict[
    IncidentType,
    set[EvidenceKind],
] = {
    IncidentType.FIRE: {
        EvidenceKind.FIRE,
        EvidenceKind.SMOKE,
        EvidenceKind.TEMPERATURE,
        EvidenceKind.GAS,
        EvidenceKind.ELECTRICAL,
    },

    IncidentType.SMOKE: {
        EvidenceKind.SMOKE,
        EvidenceKind.FIRE,
        EvidenceKind.TEMPERATURE,
        EvidenceKind.GAS,
    },

    IncidentType.THERMAL_EVENT: {
        EvidenceKind.TEMPERATURE,
        EvidenceKind.FIRE,
        EvidenceKind.SMOKE,
        EvidenceKind.ELECTRICAL,
    },

    IncidentType.ELECTRICAL_INCIDENT: {
        EvidenceKind.ELECTRICAL,
        EvidenceKind.TEMPERATURE,
        EvidenceKind.FIRE,
        EvidenceKind.SMOKE,
        EvidenceKind.ACOUSTIC,
    },

    IncidentType.GAS_LEAK: {
        EvidenceKind.GAS,
        EvidenceKind.FIRE,
        EvidenceKind.SMOKE,
        EvidenceKind.ENVIRONMENTAL,
    },

    IncidentType.LIQUID_LEAK: {
        EvidenceKind.LEAK,
        EvidenceKind.PRESSURE,
        EvidenceKind.ENVIRONMENTAL,
    },

    IncidentType.PRESSURE_INCIDENT: {
        EvidenceKind.PRESSURE,
        EvidenceKind.LEAK,
        EvidenceKind.ACOUSTIC,
    },

    IncidentType.MECHANICAL_FAILURE: {
        EvidenceKind.VIBRATION,
        EvidenceKind.ACOUSTIC,
        EvidenceKind.TEMPERATURE,
        EvidenceKind.DAMAGE,
        EvidenceKind.BEARING,
        EvidenceKind.GEAR,
    },

    IncidentType.BEARING_FAILURE_SUSPECTED: {
        EvidenceKind.BEARING,
        EvidenceKind.VIBRATION,
        EvidenceKind.TEMPERATURE,
        EvidenceKind.ACOUSTIC,
    },

    IncidentType.GEAR_FAILURE_SUSPECTED: {
        EvidenceKind.GEAR,
        EvidenceKind.VIBRATION,
        EvidenceKind.ACOUSTIC,
        EvidenceKind.DAMAGE,
    },

    IncidentType.STRUCTURAL_DAMAGE: {
        EvidenceKind.STRUCTURAL,
        EvidenceKind.DAMAGE,
        EvidenceKind.VIBRATION,
        EvidenceKind.ACOUSTIC,
    },

    IncidentType.SAFETY_ZONE_VIOLATION: {
        EvidenceKind.PERSON,
        EvidenceKind.RESTRICTED_ZONE,
        EvidenceKind.MOTION,
    },

    IncidentType.PERSON_PROXIMITY: {
        EvidenceKind.PERSON,
        EvidenceKind.PROXIMITY,
        EvidenceKind.MOTION,
    },

    IncidentType.ABNORMAL_MOTION: {
        EvidenceKind.MOTION,
        EvidenceKind.PROXIMITY,
        EvidenceKind.PERSON,
    },

    IncidentType.EQUIPMENT_DAMAGE: {
        EvidenceKind.DAMAGE,
        EvidenceKind.VIBRATION,
        EvidenceKind.TEMPERATURE,
        EvidenceKind.ELECTRICAL,
    },

    IncidentType.ENVIRONMENTAL_HAZARD: {
        EvidenceKind.ENVIRONMENTAL,
        EvidenceKind.GAS,
        EvidenceKind.LEAK,
        EvidenceKind.SMOKE,
    },
}


# ============================================================
# Fusion Engine
# ============================================================


class IncidentFusionEngine:
    """
    High-level multimodal incident fusion.

    Responsibilities
    ----------------
    * Filter unreliable evidence.
    * Deduplicate repeated detector output.
    * Correlate evidence by time / asset / zone / component.
    * Build possible incident hypotheses.
    * Rank hypotheses.
    * Calculate incident risk and confidence.
    * Determine review/escalation requirements.

    Important:
    This module does NOT execute safety-critical actions.

    It produces an incident hypothesis that should be consumed by:

        decision_engine/
            risk_policy.py
            action_policy.py
            escalation_policy.py
            intervention_policy.py

    Hard safety rules remain authoritative.
    """

    _SEVERITY_RANK = {
        IncidentSeverity.INFO: 0,
        IncidentSeverity.LOW: 1,
        IncidentSeverity.MEDIUM: 2,
        IncidentSeverity.HIGH: 3,
        IncidentSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        policy: IncidentFusionPolicy | None = None,
    ) -> None:
        self.policy = (
            policy
            or IncidentFusionPolicy()
        )

    # ========================================================
    # Public API
    # ========================================================

    def fuse(
        self,
        evidence: Iterable[IncidentEvidence],
        *,
        asset_id: str | None = None,
        site_id: str | None = None,
        zone_id: str | None = None,
        component_id: str | None = None,
    ) -> FusedIncident:

        prepared = self._prepare(
            evidence,
            asset_id=asset_id,
            site_id=site_id,
            zone_id=zone_id,
            component_id=component_id,
        )

        prepared = self._deduplicate(
            prepared
        )

        prepared = self._limit_window(
            prepared
        )

        correlation = self._correlate(
            prepared
        )

        hypotheses = self._build_hypotheses(
            prepared,
            correlation,
        )

        primary = (
            hypotheses[0]
            if hypotheses
            else None
        )

        incident_type = (
            primary.incident_type
            if primary
            else IncidentType.UNKNOWN
        )

        evidence_score = (
            self._evidence_score(
                prepared
            )
        )

        correlation_score = (
            self._correlation_score(
                correlation
            )
        )

        diversity_score = (
            self._diversity_score(
                prepared
            )
        )

        risk_score = self._risk_score(
            evidence_score,
            correlation_score,
            diversity_score,
            correlation,
        )

        confidence = self._confidence(
            prepared,
            correlation,
            primary,
        )

        severity = self._severity(
            risk_score,
            prepared,
            incident_type,
            correlation,
        )

        state = self._state(
            prepared,
            correlation,
            severity,
        )

        decision = self._decision(
            incident_type,
            severity,
            correlation,
        )

        resolved_asset = (
            asset_id
            or self._most_common(
                [
                    item.asset_id
                    for item in prepared
                    if item.asset_id
                ]
            )
        )

        resolved_site = (
            site_id
            or self._most_common(
                [
                    item.site_id
                    for item in prepared
                    if item.site_id
                ]
            )
        )

        resolved_zone = (
            zone_id
            or self._most_common(
                [
                    item.zone_id
                    for item in prepared
                    if item.zone_id
                ]
            )
        )

        resolved_component = (
            component_id
            or self._most_common(
                [
                    item.component_id
                    for item in prepared
                    if item.component_id
                ]
            )
        )

        now = (
            correlation.latest_timestamp
            or datetime.now(timezone.utc)
        )

        safety_review = (
            severity
            in {
                IncidentSeverity.HIGH,
                IncidentSeverity.CRITICAL,
            }
            and incident_type
            in {
                IncidentType.FIRE,
                IncidentType.SMOKE,
                IncidentType.THERMAL_EVENT,
                IncidentType.ELECTRICAL_INCIDENT,
                IncidentType.GAS_LEAK,
                IncidentType.PRESSURE_INCIDENT,
                IncidentType.SAFETY_ZONE_VIOLATION,
                IncidentType.PERSON_PROXIMITY,
            }
        )

        maintenance_review = (
            severity
            in {
                IncidentSeverity.MEDIUM,
                IncidentSeverity.HIGH,
                IncidentSeverity.CRITICAL,
            }
            and incident_type
            in {
                IncidentType.THERMAL_EVENT,
                IncidentType.ELECTRICAL_INCIDENT,
                IncidentType.LIQUID_LEAK,
                IncidentType.PRESSURE_INCIDENT,
                IncidentType.MECHANICAL_FAILURE,
                IncidentType.BEARING_FAILURE_SUSPECTED,
                IncidentType.GEAR_FAILURE_SUSPECTED,
                IncidentType.STRUCTURAL_DAMAGE,
                IncidentType.EQUIPMENT_DAMAGE,
            }
        )

        escalation_review = (
            severity
            == IncidentSeverity.CRITICAL
            or (
                severity == IncidentSeverity.HIGH
                and correlation.corroborated
            )
        )

        return FusedIncident(
            incident_id=str(uuid4()),
            created_at=(
                correlation.earliest_timestamp
                or now
            ),
            updated_at=now,
            incident_type=incident_type,
            state=state,
            severity=severity,
            risk_score=risk_score,
            confidence=confidence,
            asset_id=resolved_asset,
            site_id=resolved_site,
            zone_id=resolved_zone,
            component_id=resolved_component,
            evidence=prepared,
            correlation=correlation,
            hypotheses=hypotheses,
            primary_hypothesis=primary,
            decision=decision,
            requires_investigation=(
                severity
                not in {
                    IncidentSeverity.INFO,
                    IncidentSeverity.LOW,
                }
            ),
            requires_maintenance_review=(
                maintenance_review
            ),
            requires_safety_review=(
                safety_review
            ),
            requires_escalation_review=(
                escalation_review
            ),
            metadata={
                "evidence_score":
                    evidence_score,
                "correlation_score":
                    correlation_score,
                "diversity_score":
                    diversity_score,
            },
        )

    def fuse_many(
        self,
        evidence_groups: Iterable[
            Iterable[IncidentEvidence]
        ],
    ) -> list[FusedIncident]:

        return [
            self.fuse(group)
            for group in evidence_groups
        ]

    # ========================================================
    # Preparation
    # ========================================================

    def _prepare(
        self,
        evidence: Iterable[IncidentEvidence],
        *,
        asset_id: str | None,
        site_id: str | None,
        zone_id: str | None,
        component_id: str | None,
    ) -> list[IncidentEvidence]:

        output: list[
            IncidentEvidence
        ] = []

        for item in evidence:

            if (
                item.confidence
                < self.policy
                .minimum_evidence_confidence
            ):
                continue

            if (
                item.score
                < self.policy
                .minimum_evidence_score
            ):
                continue

            if (
                asset_id
                and item.asset_id
                not in {
                    None,
                    asset_id,
                }
            ):
                continue

            if (
                site_id
                and item.site_id
                not in {
                    None,
                    site_id,
                }
            ):
                continue

            if (
                zone_id
                and item.zone_id
                not in {
                    None,
                    zone_id,
                }
            ):
                continue

            if (
                component_id
                and item.component_id
                not in {
                    None,
                    component_id,
                }
            ):
                continue

            output.append(item)

        return sorted(
            output,
            key=lambda item:
                item.timestamp,
        )

    # ========================================================
    # Deduplication
    # ========================================================

    def _deduplicate(
        self,
        evidence: Sequence[
            IncidentEvidence
        ],
    ) -> list[IncidentEvidence]:

        selected: dict[
            tuple[Any, ...],
            IncidentEvidence,
        ] = {}

        for item in evidence:

            key = (
                item.source,
                item.kind,
                item.source_id,
                item.asset_id,
                item.zone_id,
                item.component_id,
            )

            existing = selected.get(key)

            if existing is None:
                selected[key] = item
                continue

            delta = abs(
                (
                    item.timestamp
                    - existing.timestamp
                ).total_seconds()
            )

            if (
                delta
                > self.policy
                .duplicate_time_seconds
            ):
                # Keep temporally distinct evidence.
                key = (
                    *key,
                    item.evidence_id,
                )

                selected[key] = item
                continue

            if (
                item.effective_score
                > existing.effective_score
            ):
                selected[key] = item

        output = list(
            selected.values()
        )

        output.sort(
            key=lambda item:
                item.timestamp
        )

        return output[
            -self.policy.max_evidence :
        ]

    # ========================================================
    # Correlation window
    # ========================================================

    def _limit_window(
        self,
        evidence: Sequence[
            IncidentEvidence
        ],
    ) -> list[IncidentEvidence]:

        if not evidence:
            return []

        latest = max(
            item.timestamp
            for item in evidence
        )

        output = [
            item
            for item in evidence
            if (
                latest - item.timestamp
            ).total_seconds()
            <= self.policy
            .correlation_window_seconds
        ]

        return output

    # ========================================================
    # Correlation
    # ========================================================

    def _correlate(
        self,
        evidence: Sequence[
            IncidentEvidence
        ],
    ) -> IncidentCorrelation:

        if not evidence:
            return IncidentCorrelation(
                evidence_count=0,
                source_count=0,
                independent_source_count=0,
                evidence_kind_count=0,
                temporal_agreement=0.0,
                spatial_agreement=0.0,
                semantic_agreement=0.0,
                confidence_agreement=0.0,
                corroborated=False,
                conflicting=False,
                earliest_timestamp=None,
                latest_timestamp=None,
                duration_seconds=0.0,
            )

        earliest = min(
            item.timestamp
            for item in evidence
        )

        latest = max(
            item.timestamp
            for item in evidence
        )

        duration = max(
            0.0,
            (
                latest - earliest
            ).total_seconds(),
        )

        temporal = exp(
            -duration
            / max(
                self.policy
                .correlation_window_seconds,
                1e-6,
            )
        )

        spatial = (
            self._spatial_agreement(
                evidence
            )
        )

        semantic = (
            self._semantic_agreement(
                evidence
            )
        )

        confidence_values = [
            item.confidence
            for item in evidence
        ]

        confidence_agreement = (
            self._agreement(
                confidence_values
            )
        )

        sources = {
            item.source
            for item in evidence
        }

        source_keys = {
            (
                item.independent_source_key
                or item.source_id
                or (
                    f"{item.source.value}:"
                    f"{item.evidence_id}"
                )
            )
            for item in evidence
        }

        kinds = {
            item.kind
            for item in evidence
        }

        independent_count = len(
            source_keys
        )

        corroborated = (
            independent_count
            >= self.policy
            .minimum_independent_sources
            and semantic >= 0.50
            and spatial >= 0.50
        )

        conflicting = (
            len(evidence) >= 2
            and (
                semantic < 0.25
                or spatial < 0.25
            )
        )

        return IncidentCorrelation(
            evidence_count=len(evidence),
            source_count=len(sources),
            independent_source_count=(
                independent_count
            ),
            evidence_kind_count=len(kinds),
            temporal_agreement=_clamp(
                temporal
            ),
            spatial_agreement=spatial,
            semantic_agreement=semantic,
            confidence_agreement=(
                confidence_agreement
            ),
            corroborated=corroborated,
            conflicting=conflicting,
            earliest_timestamp=earliest,
            latest_timestamp=latest,
            duration_seconds=duration,
        )

    # ========================================================
    # Hypotheses
    # ========================================================

    def _build_hypotheses(
        self,
        evidence: Sequence[
            IncidentEvidence
        ],
        correlation: IncidentCorrelation,
    ) -> list[IncidentHypothesis]:

        if not evidence:
            return []

        hypotheses: list[
            IncidentHypothesis
        ] = []

        for incident_type, expected_kinds in (
            _INCIDENT_RULES.items()
        ):
            supporting = [
                item
                for item in evidence
                if item.kind
                in expected_kinds
            ]

            if not supporting:
                continue

            present_kinds = {
                item.kind
                for item in supporting
            }

            coverage = (
                len(present_kinds)
                / max(
                    len(expected_kinds),
                    1,
                )
            )

            strongest = max(
                item.effective_score
                for item in supporting
            )

            average_score = mean(
                item.effective_score
                for item in supporting
            )

            independent_sources = {
                (
                    item.independent_source_key
                    or item.source_id
                    or item.evidence_id
                )
                for item in supporting
            }

            diversity = min(
                1.0,
                len(independent_sources)
                / max(
                    self.policy
                    .minimum_independent_sources,
                    1,
                ),
            )

            score = (
                strongest * 0.45
                + average_score * 0.25
                + coverage * 0.15
                + diversity * 0.15
            )

            if correlation.corroborated:
                score += 0.05

            score = _clamp(score)

            confidence = _clamp(
                mean(
                    item.confidence
                    for item in supporting
                )
                * 0.80
                + diversity * 0.20
            )

            severity = self._score_severity(
                score
            )

            rationale = [
                (
                    f"{len(supporting)} supporting "
                    "evidence item(s)"
                ),
                (
                    f"{len(present_kinds)} evidence "
                    "kind(s)"
                ),
                (
                    f"{len(independent_sources)} "
                    "independent source(s)"
                ),
            ]

            hypotheses.append(
                IncidentHypothesis(
                    hypothesis_id=str(
                        uuid4()
                    ),
                    incident_type=(
                        incident_type
                    ),
                    score=score,
                    confidence=confidence,
                    severity=severity,
                    supporting_evidence_ids=[
                        item.evidence_id
                        for item in supporting
                    ],
                    rationale=rationale,
                )
            )

        hypotheses.sort(
            key=lambda item: (
                item.score,
                item.confidence,
            ),
            reverse=True,
        )

        if not hypotheses:
            hypotheses.append(
                IncidentHypothesis(
                    hypothesis_id=str(
                        uuid4()
                    ),
                    incident_type=(
                        IncidentType
                        .MULTI_SIGNAL_HAZARD
                        if len(evidence) > 1
                        else IncidentType.UNKNOWN
                    ),
                    score=self._evidence_score(
                        evidence
                    ),
                    confidence=mean(
                        item.confidence
                        for item in evidence
                    ),
                    severity=(
                        self._score_severity(
                            self._evidence_score(
                                evidence
                            )
                        )
                    ),
                    supporting_evidence_ids=[
                        item.evidence_id
                        for item in evidence
                    ],
                    rationale=[
                        (
                            "Evidence did not map "
                            "cleanly to a specific "
                            "incident class"
                        )
                    ],
                )
            )

        return hypotheses

    # ========================================================
    # Scores
    # ========================================================

    @staticmethod
    def _evidence_score(
        evidence: Sequence[
            IncidentEvidence
        ],
    ) -> float:

        if not evidence:
            return 0.0

        scores = sorted(
            (
                item.effective_score
                for item in evidence
            ),
            reverse=True,
        )

        strongest = scores[0]

        if len(scores) == 1:
            return strongest

        supporting = mean(
            scores[1:]
        )

        return _clamp(
            strongest * 0.75
            + supporting * 0.25
        )

    def _correlation_score(
        self,
        correlation: IncidentCorrelation,
    ) -> float:

        p = self.policy

        return _clamp(
            correlation.temporal_agreement
            * p.temporal_weight
            + correlation.spatial_agreement
            * p.spatial_weight
            + correlation.semantic_agreement
            * p.semantic_weight
            + correlation.confidence_agreement
            * p.confidence_weight
        )

    def _diversity_score(
        self,
        evidence: Sequence[
            IncidentEvidence
        ],
    ) -> float:

        if not evidence:
            return 0.0

        sources = {
            (
                item.independent_source_key
                or item.source_id
                or item.evidence_id
            )
            for item in evidence
        }

        kinds = {
            item.kind
            for item in evidence
        }

        source_score = min(
            1.0,
            len(sources) / 4.0,
        )

        kind_score = min(
            1.0,
            len(kinds) / 4.0,
        )

        return _clamp(
            source_score * 0.60
            + kind_score * 0.40
        )

    def _risk_score(
        self,
        evidence_score: float,
        correlation_score: float,
        diversity_score: float,
        correlation: IncidentCorrelation,
    ) -> float:

        p = self.policy

        score = (
            evidence_score
            * p.evidence_weight
            + correlation_score
            * p.correlation_weight
            + diversity_score
            * p.diversity_weight
        )

        if correlation.corroborated:
            score += (
                p.corroboration_bonus
            )

        if correlation.conflicting:
            score -= (
                p.conflict_penalty
            )

        return _clamp(score)

    # ========================================================
    # Confidence
    # ========================================================

    def _confidence(
        self,
        evidence: Sequence[
            IncidentEvidence
        ],
        correlation: IncidentCorrelation,
        primary: IncidentHypothesis | None,
    ) -> float:

        if not evidence:
            return 0.0

        base = mean(
            item.confidence
            for item in evidence
        )

        correlation_score = (
            self._correlation_score(
                correlation
            )
        )

        hypothesis_confidence = (
            primary.confidence
            if primary
            else 0.0
        )

        confidence = (
            base * 0.50
            + correlation_score * 0.25
            + hypothesis_confidence * 0.25
        )

        if correlation.corroborated:
            confidence += 0.05

        if correlation.conflicting:
            confidence -= 0.15

        return _clamp(confidence)

    # ========================================================
    # Severity
    # ========================================================

    def _severity(
        self,
        risk_score: float,
        evidence: Sequence[
            IncidentEvidence
        ],
        incident_type: IncidentType,
        correlation: IncidentCorrelation,
    ) -> IncidentSeverity:

        strongest_input = max(
            (
                self._SEVERITY_RANK[
                    item.severity
                ]
                for item in evidence
            ),
            default=0,
        )

        score_severity = (
            self._score_severity(
                risk_score
            )
        )

        score_rank = (
            self._SEVERITY_RANK[
                score_severity
            ]
        )

        final_rank = max(
            strongest_input,
            score_rank,
        )

        # Corroborated safety-critical evidence can
        # promote a HIGH hypothesis to CRITICAL review.
        if (
            correlation.corroborated
            and risk_score >= 0.85
            and incident_type
            in {
                IncidentType.FIRE,
                IncidentType.GAS_LEAK,
                IncidentType.PERSON_PROXIMITY,
                IncidentType.SAFETY_ZONE_VIOLATION,
            }
        ):
            final_rank = max(
                final_rank,
                self._SEVERITY_RANK[
                    IncidentSeverity.CRITICAL
                ],
            )

        return self._severity_from_rank(
            final_rank
        )

    def _score_severity(
        self,
        score: float,
    ) -> IncidentSeverity:

        p = self.policy

        if score >= p.critical_threshold:
            return IncidentSeverity.CRITICAL

        if score >= p.high_threshold:
            return IncidentSeverity.HIGH

        if score >= p.medium_threshold:
            return IncidentSeverity.MEDIUM

        if score >= p.low_threshold:
            return IncidentSeverity.LOW

        return IncidentSeverity.INFO

    def _severity_from_rank(
        self,
        rank: int,
    ) -> IncidentSeverity:

        for severity, value in (
            self._SEVERITY_RANK.items()
        ):
            if value == rank:
                return severity

        return IncidentSeverity.INFO

    # ========================================================
    # State / Decision
    # ========================================================

    @staticmethod
    def _state(
        evidence: Sequence[
            IncidentEvidence
        ],
        correlation: IncidentCorrelation,
        severity: IncidentSeverity,
    ) -> IncidentState:

        if (
            severity
            == IncidentSeverity.CRITICAL
        ):
            return IncidentState.CRITICAL

        if correlation.conflicting:
            return IncidentState.CONFLICTING

        if correlation.corroborated:
            return IncidentState.CORROBORATED

        if len(evidence) >= 2:
            return IncidentState.SUSPECTED

        return IncidentState.CANDIDATE

    @staticmethod
    def _decision(
        incident_type: IncidentType,
        severity: IncidentSeverity,
        correlation: IncidentCorrelation,
    ) -> FusionDecision:

        if severity == IncidentSeverity.INFO:
            return FusionDecision.IGNORE

        if severity == IncidentSeverity.LOW:
            return FusionDecision.MONITOR

        safety_types = {
            IncidentType.FIRE,
            IncidentType.SMOKE,
            IncidentType.THERMAL_EVENT,
            IncidentType.ELECTRICAL_INCIDENT,
            IncidentType.GAS_LEAK,
            IncidentType.PRESSURE_INCIDENT,
            IncidentType.SAFETY_ZONE_VIOLATION,
            IncidentType.PERSON_PROXIMITY,
        }

        maintenance_types = {
            IncidentType.MECHANICAL_FAILURE,
            IncidentType.BEARING_FAILURE_SUSPECTED,
            IncidentType.GEAR_FAILURE_SUSPECTED,
            IncidentType.STRUCTURAL_DAMAGE,
            IncidentType.EQUIPMENT_DAMAGE,
            IncidentType.LIQUID_LEAK,
        }

        if (
            severity == IncidentSeverity.CRITICAL
            and correlation.corroborated
        ):
            return (
                FusionDecision
                .ESCALATION_REVIEW
            )

        if incident_type in safety_types:
            return FusionDecision.SAFETY_REVIEW

        if incident_type in maintenance_types:
            return (
                FusionDecision
                .MAINTENANCE_REVIEW
            )

        return FusionDecision.INVESTIGATE

    # ========================================================
    # Spatial agreement
    # ========================================================

    @staticmethod
    def _spatial_agreement(
        evidence: Sequence[
            IncidentEvidence
        ],
    ) -> float:

        if len(evidence) <= 1:
            return 0.50

        scores: list[float] = []

        for i in range(
            len(evidence)
        ):
            for j in range(
                i + 1,
                len(evidence),
            ):
                first = evidence[i]
                second = evidence[j]

                pair_scores: list[
                    float
                ] = []

                pair_scores.append(
                    _identity_agreement(
                        first.asset_id,
                        second.asset_id,
                    )
                )

                pair_scores.append(
                    _identity_agreement(
                        first.zone_id,
                        second.zone_id,
                    )
                )

                if (
                    first.component_id
                    or second.component_id
                ):
                    pair_scores.append(
                        _identity_agreement(
                            first.component_id,
                            second.component_id,
                        )
                    )

                scores.append(
                    mean(pair_scores)
                )

        return _clamp(
            mean(scores)
            if scores
            else 0.50
        )

    # ========================================================
    # Semantic agreement
    # ========================================================

    @staticmethod
    def _semantic_agreement(
        evidence: Sequence[
            IncidentEvidence
        ],
    ) -> float:

        if not evidence:
            return 0.0

        kinds = {
            item.kind
            for item in evidence
        }

        best = 0.0

        for expected in (
            _INCIDENT_RULES.values()
        ):
            overlap = len(
                kinds & expected
            )

            if overlap == 0:
                continue

            # Measure how much of observed evidence
            # belongs to the same incident family.
            precision = (
                overlap
                / max(
                    len(kinds),
                    1,
                )
            )

            coverage = (
                overlap
                / max(
                    min(
                        len(expected),
                        3,
                    ),
                    1,
                )
            )

            score = (
                precision * 0.65
                + min(
                    coverage,
                    1.0,
                ) * 0.35
            )

            best = max(
                best,
                score,
            )

        return _clamp(best)

    # ========================================================
    # Agreement helper
    # ========================================================

    @staticmethod
    def _agreement(
        values: Sequence[float],
    ) -> float:

        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        avg = mean(values)

        deviation = mean(
            abs(value - avg)
            for value in values
        )

        return _clamp(
            1.0 - deviation
        )

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _most_common(
        values: Sequence[str],
    ) -> str | None:

        if not values:
            return None

        counts: dict[
            str,
            int,
        ] = {}

        for value in values:
            counts[value] = (
                counts.get(
                    value,
                    0,
                )
                + 1
            )

        return max(
            counts,
            key=counts.get,  # type: ignore[arg-type]
        )


# ============================================================
# Convenience Constructor
# ============================================================


def create_incident_evidence(
    kind: EvidenceKind,
    score: float,
    confidence: float,
    *,
    source: IncidentSource,
    asset_id: str | None = None,
    site_id: str | None = None,
    zone_id: str | None = None,
    component_id: str | None = None,
    source_id: str | None = None,
    independent_source_key: str | None = None,
    severity: IncidentSeverity = (
        IncidentSeverity.LOW
    ),
    description: str = "",
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> IncidentEvidence:

    return IncidentEvidence(
        evidence_id=str(uuid4()),
        timestamp=(
            timestamp
            or datetime.now(
                timezone.utc
            )
        ),
        source=source,
        kind=kind,
        score=score,
        confidence=confidence,
        asset_id=asset_id,
        site_id=site_id,
        zone_id=zone_id,
        component_id=component_id,
        source_id=source_id,
        severity=severity,
        description=description,
        independent_source_key=(
            independent_source_key
        ),
        metadata=metadata or {},
    )


# ============================================================
# Common Evidence Adapters
# ============================================================


def temperature_evidence(
    anomaly_score: float,
    confidence: float,
    *,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    zone_id: str | None = None,
    severity: IncidentSeverity = (
        IncidentSeverity.MEDIUM
    ),
    timestamp: datetime | None = None,
) -> IncidentEvidence:

    return create_incident_evidence(
        EvidenceKind.TEMPERATURE,
        anomaly_score,
        confidence,
        source=(
            IncidentSource
            .ANOMALY_DETECTOR
        ),
        asset_id=asset_id,
        component_id=component_id,
        zone_id=zone_id,
        source_id=sensor_id,
        independent_source_key=(
            f"temperature:{sensor_id}"
            if sensor_id
            else None
        ),
        severity=severity,
        description=(
            "Temperature anomaly evidence"
        ),
        timestamp=timestamp,
    )


def vibration_evidence(
    anomaly_score: float,
    confidence: float,
    *,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    zone_id: str | None = None,
    severity: IncidentSeverity = (
        IncidentSeverity.MEDIUM
    ),
    timestamp: datetime | None = None,
) -> IncidentEvidence:

    return create_incident_evidence(
        EvidenceKind.VIBRATION,
        anomaly_score,
        confidence,
        source=(
            IncidentSource
            .ANOMALY_DETECTOR
        ),
        asset_id=asset_id,
        component_id=component_id,
        zone_id=zone_id,
        source_id=sensor_id,
        independent_source_key=(
            f"vibration:{sensor_id}"
            if sensor_id
            else None
        ),
        severity=severity,
        description=(
            "Vibration anomaly evidence"
        ),
        timestamp=timestamp,
    )


def electrical_evidence(
    anomaly_score: float,
    confidence: float,
    *,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    zone_id: str | None = None,
    severity: IncidentSeverity = (
        IncidentSeverity.MEDIUM
    ),
    timestamp: datetime | None = None,
) -> IncidentEvidence:

    return create_incident_evidence(
        EvidenceKind.ELECTRICAL,
        anomaly_score,
        confidence,
        source=(
            IncidentSource
            .ANOMALY_DETECTOR
        ),
        asset_id=asset_id,
        component_id=component_id,
        zone_id=zone_id,
        source_id=sensor_id,
        independent_source_key=(
            f"electrical:{sensor_id}"
            if sensor_id
            else None
        ),
        severity=severity,
        description=(
            "Electrical anomaly evidence"
        ),
        timestamp=timestamp,
    )


def visual_evidence(
    kind: EvidenceKind,
    score: float,
    confidence: float,
    *,
    camera_id: str,
    asset_id: str | None = None,
    zone_id: str | None = None,
    severity: IncidentSeverity = (
        IncidentSeverity.MEDIUM
    ),
    thermal_camera: bool = False,
    timestamp: datetime | None = None,
) -> IncidentEvidence:

    return create_incident_evidence(
        kind,
        score,
        confidence,
        source=(
            IncidentSource.THERMAL_CAMERA
            if thermal_camera
            else IncidentSource.COMPUTER_VISION
        ),
        asset_id=asset_id,
        zone_id=zone_id,
        source_id=camera_id,
        independent_source_key=(
            f"camera:{camera_id}"
        ),
        severity=severity,
        description=(
            "Visual incident evidence"
        ),
        timestamp=timestamp,
    )


# ============================================================
# Utility Functions
# ============================================================


def _identity_agreement(
    first: str | None,
    second: str | None,
) -> float:

    if (
        first is None
        or second is None
    ):
        return 0.50

    return (
        1.0
        if first == second
        else 0.0
    )


def _valid_number(
    value: Any,
) -> bool:

    try:
        return isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    if not _valid_number(value):
        return minimum

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# ============================================================
# Default Engine
# ============================================================


default_incident_fusion_engine = (
    IncidentFusionEngine()
)


__all__ = [
    "IncidentType",
    "IncidentSeverity",
    "IncidentState",
    "IncidentSource",
    "EvidenceKind",
    "FusionDecision",
    "IncidentEvidence",
    "IncidentCorrelation",
    "IncidentHypothesis",
    "FusedIncident",
    "IncidentFusionPolicy",
    "IncidentFusionEngine",
    "create_incident_evidence",
    "temperature_evidence",
    "vibration_evidence",
    "electrical_evidence",
    "visual_evidence",
    "default_incident_fusion_engine",
