
"""
app/ai/cv/visual_health_score.py

Industrial Visual Asset Health Scoring
======================================

Python 3.11+

Features:
- Multi-detector inspection aggregation
- Crack, corrosion and damage assessment
- Leakage and overheating assessment
- Weighted visual health scoring
- Severity-based risk penalties
- Missing-inspection coverage reporting
- Critical finding identification
- Explainable score breakdown
- Maintenance review recommendations
- Tenant and asset metadata
- JSON-compatible output
- Batch and async scoring

Dependencies:
    Python standard library only

IMPORTANT:
    This is an explainable screening score,
    not a validated probability of failure,
    remaining useful life estimate, or
    engineering fitness-for-service assessment.

    Detector severity labels are preliminary.
    Visual inspections may miss hidden defects.

    Never use this score alone to authorize
    operation of safety-critical equipment.
"""

from __future__ import annotations

import asyncio
import math
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class VisualHealthScoreError(Exception):
    """Base visual health scoring exception."""


class VisualHealthValidationError(
    VisualHealthScoreError
):
    """Invalid configuration or inspection data."""


# ============================================================
# ENUMS
# ============================================================

class VisualInspectionType(str, Enum):
    CRACK = "crack"
    CORROSION = "corrosion"
    DAMAGE = "damage"
    LEAKAGE = "leakage"
    OVERHEATING = "overheating"


class VisualHealthStatus(str, Enum):
    GOOD = "good"
    WATCH = "watch"
    DEGRADED = "degraded"
    POOR = "poor"
    CRITICAL = "critical"
    NOT_ASSESSED = "not_assessed"


class InspectionSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTENSIVE = "extensive"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class VisualHealthScoreConfig:
    """
    Default weights are illustrative.

    Adjust them using equipment-specific
    engineering assessments and validation.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "crack": 0.25,
            "corrosion": 0.20,
            "damage": 0.15,
            "leakage": 0.20,
            "overheating": 0.20,
        }
    )

    severity_penalties: dict[str, float] = field(
        default_factory=lambda: {
            "none": 0.0,
            "low": 15.0,
            "moderate": 40.0,
            "high": 70.0,
            "extensive": 90.0,
            "critical": 100.0,
        }
    )

    # These are heuristic score bands.
    good_min_score: float = 85.0
    watch_min_score: float = 70.0
    degraded_min_score: float = 50.0
    poor_min_score: float = 25.0

    # Maximum inspected area contribution
    # to a component penalty.
    area_penalty_multiplier: float = 0.30

    # Additional penalty for multiple findings.
    finding_count_penalty: float = 2.0

    # A critical finding may cap the score,
    # but does not replace engineering review.
    critical_score_cap: float = 25.0

    # Score coverage threshold for displaying
    # a definitive-looking status.
    minimum_coverage_percent: float = 80.0

    def __post_init__(self) -> None:
        expected = {
            item.value
            for item in VisualInspectionType
        }

        if set(self.weights) != expected:
            raise VisualHealthValidationError(
                "Weights must contain exactly: "
                + ", ".join(sorted(expected))
            )

        for name, weight in self.weights.items():
            if (
                not math.isfinite(weight)
                or weight < 0
            ):
                raise VisualHealthValidationError(
                    f"Invalid weight: {name}"
                )

        if sum(self.weights.values()) <= 0:
            raise VisualHealthValidationError(
                "Total inspection weight must "
                "be greater than zero"
            )

        expected_severities = {
            item.value
            for item in InspectionSeverity
        }

        if (
            set(self.severity_penalties)
            != expected_severities
        ):
            raise VisualHealthValidationError(
                "Invalid severity penalty mapping"
            )

        for value in self.severity_penalties.values():
            if (
                not math.isfinite(value)
                or not 0 <= value <= 100
            ):
                raise VisualHealthValidationError(
                    "Severity penalties must "
                    "be between 0 and 100"
                )

        if not (
            0
            <= self.poor_min_score
            < self.degraded_min_score
            < self.watch_min_score
            < self.good_min_score
            <= 100
        ):
            raise VisualHealthValidationError(
                "Score thresholds must increase"
            )

        if not (
            0 <= self.area_penalty_multiplier <= 1
        ):
            raise VisualHealthValidationError(
                "Invalid area penalty multiplier"
            )

        if self.finding_count_penalty < 0:
            raise VisualHealthValidationError(
                "finding_count_penalty cannot "
                "be negative"
            )

        if not (
            0 <= self.critical_score_cap <= 100
        ):
            raise VisualHealthValidationError(
                "Invalid critical score cap"
            )

        if not (
            0
            <= self.minimum_coverage_percent
            <= 100
        ):
            raise VisualHealthValidationError(
                "Invalid minimum coverage"
            )


# ============================================================
# INPUT SCHEMA
# ============================================================

@dataclass(slots=True)
class VisualInspectionInput:
    """
    Normalized detector result.

    detected:
        True  = finding observed
        False = inspection completed,
                no finding observed

    severity:
        Detector-reported screening severity.

    affected_area_percent:
        Optional affected image-area percentage.

    finding_count:
        Number of detected regions.

    inspected_at:
        Optional timestamp for freshness review.
    """

    inspection_type: VisualInspectionType
    detected: bool
    severity: InspectionSeverity

    affected_area_percent: float = 0.0
    finding_count: int = 0

    inspection_id: str | None = None
    inspected_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if isinstance(self.inspection_type, str):
            self.inspection_type = (
                VisualInspectionType(
                    self.inspection_type
                )
            )

        if isinstance(self.severity, str):
            self.severity = InspectionSeverity(
                self.severity
            )

        if not isinstance(self.detected, bool):
            raise VisualHealthValidationError(
                "detected must be boolean"
            )

        if (
            not math.isfinite(
                self.affected_area_percent
            )
            or not (
                0
                <= self.affected_area_percent
                <= 100
            )
        ):
            raise VisualHealthValidationError(
                "Invalid affected_area_percent"
            )

        if (
            isinstance(self.finding_count, bool)
            or not isinstance(self.finding_count, int)
            or self.finding_count < 0
        ):
            raise VisualHealthValidationError(
                "finding_count must be "
                "a non-negative integer"
            )

        if (
            self.inspected_at is not None
            and self.inspected_at.tzinfo is None
        ):
            raise VisualHealthValidationError(
                "inspected_at must be timezone-aware"
            )

        if not self.detected:
            if (
                self.severity
                != InspectionSeverity.NONE
                or self.finding_count != 0
                or self.affected_area_percent != 0
            ):
                raise VisualHealthValidationError(
                    "No-detection inspections must "
                    "have zero findings, zero area "
                    "and severity NONE"
                )

        elif (
            self.severity == InspectionSeverity.NONE
        ):
            raise VisualHealthValidationError(
                "Detected findings require "
                "a non-NONE severity"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_type": (
                self.inspection_type.value
            ),
            "detected": self.detected,
            "severity": self.severity.value,
            "affected_area_percent": (
                self.affected_area_percent
            ),
            "finding_count": self.finding_count,
            "inspection_id": self.inspection_id,
            "inspected_at": (
                self.inspected_at.isoformat()
                if self.inspected_at
                else None
            ),
            "metadata": dict(self.metadata),
        }


# ============================================================
# COMPONENT SCORE
# ============================================================

@dataclass(slots=True)
class VisualComponentScore:
    inspection_type: str

    assessed: bool
    detected: bool | None

    severity: str | None

    component_score: float | None
    component_penalty: float | None

    weight: float
    weighted_penalty: float | None

    affected_area_percent: float | None
    finding_count: int | None

    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_type": self.inspection_type,
            "assessed": self.assessed,
            "detected": self.detected,
            "severity": self.severity,
            "component_score": (
                self.component_score
            ),
            "component_penalty": (
                self.component_penalty
            ),
            "weight": self.weight,
            "weighted_penalty": (
                self.weighted_penalty
            ),
            "affected_area_percent": (
                self.affected_area_percent
            ),
            "finding_count": self.finding_count,
            "explanation": self.explanation,
        }


# ============================================================
# FINAL RESULT
# ============================================================

@dataclass(slots=True)
class VisualHealthScoreResult:
    assessment_id: str

    company_id: str
    asset_id: str
    equipment_id: str | None

    visual_health_score: float | None
    health_status: str

    inspection_coverage_percent: float
    coverage_sufficient: bool

    assessed_inspection_types: list[str]
    missing_inspection_types: list[str]

    component_scores: list[VisualComponentScore]

    critical_findings: list[str]
    recommendations: list[str]

    requires_human_review: bool
    safety_escalation_recommended: bool

    assessed_at: datetime

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "company_id": self.company_id,
            "asset_id": self.asset_id,
            "equipment_id": self.equipment_id,
            "visual_health_score": (
                self.visual_health_score
            ),
            "health_status": self.health_status,
            "inspection_coverage_percent": (
                self.inspection_coverage_percent
            ),
            "coverage_sufficient": (
                self.coverage_sufficient
            ),
            "assessed_inspection_types": list(
                self.assessed_inspection_types
            ),
            "missing_inspection_types": list(
                self.missing_inspection_types
            ),
            "component_scores": [
                item.to_dict()
                for item in self.component_scores
            ],
            "critical_findings": list(
                self.critical_findings
            ),
            "recommendations": list(
                self.recommendations
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "safety_escalation_recommended": (
                self.safety_escalation_recommended
            ),
            "assessed_at": (
                self.assessed_at.isoformat()
            ),
            "metadata": dict(self.metadata),
        }


# ============================================================
# DETECTOR RESULT ADAPTER
# ============================================================

class VisualInspectionAdapter:
    """
    Converts existing detector dataclasses or
    result dictionaries to normalized inputs.

    Supported detector result formats:
    - CrackDetectionResult
    - CorrosionDetectionResult
    - DamageDetectionResult
    - LeakageDetectionResult
    - OverheatingDetectionResult

    The adapter does not independently verify
    detector accuracy or inspection quality.
    """

    @staticmethod
    def _as_mapping(
        result: Any,
    ) -> Mapping[str, Any]:
        if isinstance(result, Mapping):
            return result

        if hasattr(result, "to_dict"):
            data = result.to_dict()

            if isinstance(data, Mapping):
                return data

        raise VisualHealthValidationError(
            "Detector result must be a mapping "
            "or expose to_dict()"
        )

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            parsed = value

        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise VisualHealthValidationError(
                    "Invalid inspection timestamp"
                ) from exc

        else:
            raise VisualHealthValidationError(
                "Unsupported inspection timestamp"
            )

        if parsed.tzinfo is None:
            raise VisualHealthValidationError(
                "Inspection timestamp must "
                "include a timezone"
            )

        return parsed

    @staticmethod
    def _severity(
        value: Any,
        *,
        detected: bool,
    ) -> InspectionSeverity:
        if not detected:
            return InspectionSeverity.NONE

        if isinstance(value, Enum):
            value = value.value

        if value is None:
            return InspectionSeverity.LOW

        try:
            severity = InspectionSeverity(
                str(value).lower()
            )
        except ValueError as exc:
            raise VisualHealthValidationError(
                f"Unknown severity: {value}"
            ) from exc

        if severity == InspectionSeverity.NONE:
            return InspectionSeverity.LOW

        return severity

    @classmethod
    def from_detector(
        cls,
        inspection_type: (
            VisualInspectionType | str
        ),
        result: Any,
    ) -> VisualInspectionInput:
        kind = VisualInspectionType(
            inspection_type
        )

        data = cls._as_mapping(
            result
        )

        if kind == VisualInspectionType.CRACK:
            detected_key = "detected"
            area_key = None
            count_key = "crack_count"
            severity_key = "severity"

        elif kind == VisualInspectionType.CORROSION:
            detected_key = "detected"
            area_key = "corrosion_area_percent"
            count_key = None
            severity_key = "severity"

        elif kind == VisualInspectionType.DAMAGE:
            detected_key = "detected"
            area_key = "damage_area_percent"
            count_key = "damage_region_count"
            severity_key = "severity"

        elif kind == VisualInspectionType.LEAKAGE:
            detected_key = (
                "leakage_candidate_detected"
            )
            area_key = "affected_area_percent"
            count_key = None
            severity_key = "severity"

        else:
            detected_key = "overheating_detected"
            area_key = "hotspot_area_percent"
            count_key = None
            severity_key = "severity"

        if detected_key not in data:
            raise VisualHealthValidationError(
                f"Missing detector field: {detected_key}"
            )

        detected = data[detected_key]

        if not isinstance(detected, bool):
            raise VisualHealthValidationError(
                "Detector detection flag "
                "must be boolean"
            )

        if area_key is not None:
            area = float(
                data.get(area_key, 0.0)
            )
        else:
            area = 0.0

        if count_key is not None:
            count = int(
                data.get(count_key, 0)
            )
        else:
            regions = data.get(
                "regions",
                data.get("hotspots", []),
            )
            count = len(regions)

        if not detected:
            area = 0.0
            count = 0

        severity = cls._severity(
            data.get(severity_key),
            detected=detected,
        )

        return VisualInspectionInput(
            inspection_type=kind,
            detected=detected,
            severity=severity,
            affected_area_percent=area,
            finding_count=count,
            inspection_id=data.get(
                "inspection_id"
            ),
            inspected_at=cls._parse_timestamp(
                data.get("inspected_at")
            ),
            metadata={
                "source": "detector_result",
                "detection_mode": data.get(
                    "detection_mode"
                ),
                "requires_human_review": (
                    data.get(
                        "requires_human_review"
                    )
                ),
            },
        )


# ============================================================
# HEALTH SCORE ENGINE
# ============================================================

class VisualHealthScoreEngine:
    """
    Explainable visual condition scoring.

    The weighted score is normalized over
    available inspection types.

    Missing inspections reduce coverage,
    not the score directly.

    If coverage is insufficient, health status
    is NOT_ASSESSED even when a provisional
    numerical score can be calculated.
    """

    def __init__(
        self,
        config: (
            VisualHealthScoreConfig | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or VisualHealthScoreConfig()
        )

    @staticmethod
    def _validate_identity(
        company_id: str,
        asset_id: str,
    ) -> None:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise VisualHealthValidationError(
                "company_id is required"
            )

        if (
            not isinstance(asset_id, str)
            or not asset_id.strip()
        ):
            raise VisualHealthValidationError(
                "asset_id is required"
            )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 100.0,
    ) -> float:
        return max(
            minimum,
            min(maximum, value),
        )

    # ========================================================
    # COMPONENT PENALTY
    # ========================================================

    def _component_penalty(
        self,
        inspection: VisualInspectionInput,
    ) -> float:
        if not inspection.detected:
            return 0.0

        base = (
            self.config.severity_penalties[
                inspection.severity.value
            ]
        )

        # Scale area contribution from
        # 0 to 100 percentage points.
        area_penalty = (
            inspection.affected_area_percent
            * self.config.area_penalty_multiplier
        )

        count_penalty = (
            max(
                0,
                inspection.finding_count - 1,
            )
            * self.config.finding_count_penalty
        )

        return self._clamp(
            base
            + area_penalty
            + count_penalty
        )

    # ========================================================
    # SCORE CLASSIFICATION
    # ========================================================

    def _classify_status(
        self,
        score: float,
    ) -> VisualHealthStatus:
        if score >= self.config.good_min_score:
            return VisualHealthStatus.GOOD

        if score >= self.config.watch_min_score:
            return VisualHealthStatus.WATCH

        if (
            score
            >= self.config.degraded_min_score
        ):
            return VisualHealthStatus.DEGRADED

        if score >= self.config.poor_min_score:
            return VisualHealthStatus.POOR

        return VisualHealthStatus.CRITICAL

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    @staticmethod
    def _recommendations(
        inspections: Mapping[
            str,
            VisualInspectionInput,
        ],
        missing: list[str],
        critical_findings: list[str],
    ) -> list[str]:
        recommendations: list[str] = []

        messages = {
            "crack": (
                "Arrange qualified inspection "
                "of detected cracks and assess "
                "structural significance."
            ),
            "corrosion": (
                "Inspect corrosion extent and "
                "verify remaining material "
                "condition using appropriate "
                "measurement methods."
            ),
            "damage": (
                "Review detected surface damage "
                "and determine whether repair "
                "or further inspection is needed."
            ),
            "leakage": (
                "Investigate visible leakage "
                "candidates and verify using "
                "appropriate physical inspection "
                "and leak-detection instruments."
            ),
            "overheating": (
                "Verify thermal anomalies using "
                "calibrated instruments and "
                "equipment-specific temperature "
                "limits."
            ),
        }

        for kind, inspection in inspections.items():
            if inspection.detected:
                recommendations.append(
                    messages[kind]
                )

        if critical_findings:
            recommendations.append(
                "Prioritize qualified safety "
                "and maintenance review of "
                "high-severity findings."
            )

        if missing:
            recommendations.append(
                "Complete missing inspection "
                "categories before relying on "
                "the overall visual score: "
                + ", ".join(missing)
                + "."
            )

        if not recommendations:
            recommendations.append(
                "No findings were reported by "
                "the supplied inspections. "
                "Continue scheduled maintenance "
                "and condition monitoring."
            )

        return recommendations

    # ========================================================
    # CALCULATE SCORE
    # ========================================================

    def calculate(
        self,
        inspections: Sequence[
            VisualInspectionInput
        ],
        *,
        company_id: str,
        asset_id: str,
        equipment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VisualHealthScoreResult:
        self._validate_identity(
            company_id,
            asset_id,
        )

        by_type: dict[
            str,
            VisualInspectionInput,
        ] = {}

        for inspection in inspections:
            if not isinstance(
                inspection,
                VisualInspectionInput,
            ):
                raise VisualHealthValidationError(
                    "Expected VisualInspectionInput"
                )

            kind = (
                inspection.inspection_type.value
            )

            if kind in by_type:
                raise VisualHealthValidationError(
                    "Duplicate inspection type: "
                    + kind
                )

            by_type[kind] = inspection

        all_types = [
            kind.value
            for kind in VisualInspectionType
        ]

        assessed = [
            kind
            for kind in all_types
            if kind in by_type
        ]

        missing = [
            kind
            for kind in all_types
            if kind not in by_type
        ]

        total_weight = sum(
            self.config.weights.values()
        )

        assessed_weight = sum(
            self.config.weights[kind]
            for kind in assessed
        )

        coverage = (
            assessed_weight
            / total_weight
            * 100.0
        )

        coverage_sufficient = (
            coverage
            >= self.config.minimum_coverage_percent
        )

        components: list[
            VisualComponentScore
        ] = []

        weighted_penalty_sum = 0.0

        critical_findings: list[str] = []

        for kind in all_types:
            weight = self.config.weights[kind]

            inspection = by_type.get(
                kind
            )

            if inspection is None:
                components.append(
                    VisualComponentScore(
                        inspection_type=kind,
                        assessed=False,
                        detected=None,
                        severity=None,
                        component_score=None,
                        component_penalty=None,
                        weight=weight,
                        weighted_penalty=None,
                        affected_area_percent=None,
                        finding_count=None,
                        explanation=(
                            "Inspection not supplied; "
                            "condition unknown."
                        ),
                    )
                )

                continue

            penalty = (
                self._component_penalty(
                    inspection
                )
            )

            component_score = (
                100.0 - penalty
            )

            weighted_penalty = (
                penalty * weight
            )

            weighted_penalty_sum += (
                weighted_penalty
            )

            if (
                inspection.detected
                and inspection.severity
                in {
                    InspectionSeverity.HIGH,
                    InspectionSeverity.EXTENSIVE,
                    InspectionSeverity.CRITICAL,
                }
            ):
                critical_findings.append(
                    kind
                )

            components.append(
                VisualComponentScore(
                    inspection_type=kind,
                    assessed=True,
                    detected=inspection.detected,
                    severity=(
                        inspection.severity.value
                    ),
                    component_score=round(
                        component_score,
                        2,
                    ),
                    component_penalty=round(
                        penalty,
                        2,
                    ),
                    weight=weight,
                    weighted_penalty=round(
                        weighted_penalty,
                        2,
                    ),
                    affected_area_percent=(
                        inspection
                        .affected_area_percent
                    ),
                    finding_count=(
                        inspection.finding_count
                    ),
                    explanation=(
                        f"Severity: "
                        f"{inspection.severity.value}; "
                        f"area: "
                        f"{inspection.affected_area_percent:.2f}%; "
                        f"findings: "
                        f"{inspection.finding_count}."
                    ),
                )
            )

        if assessed_weight > 0:
            normalized_penalty = (
                weighted_penalty_sum
                / assessed_weight
            )

            score: float | None = (
                self._clamp(
                    100.0 - normalized_penalty
                )
            )

            if critical_findings:
                score = min(
                    score,
                    self.config.critical_score_cap,
                )

            score = round(
                score,
                2,
            )

        else:
            score = None

        if (
            score is None
            or not coverage_sufficient
        ):
            status = (
                VisualHealthStatus
                .NOT_ASSESSED
            )

        else:
            status = self._classify_status(
                score
            )

        any_findings = any(
            inspection.detected
            for inspection in by_type.values()
        )

        recommendations = (
            self._recommendations(
                by_type,
                missing,
                critical_findings,
            )
        )

        return VisualHealthScoreResult(
            assessment_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            visual_health_score=score,
            health_status=status.value,
            inspection_coverage_percent=round(
                coverage,
                2,
            ),
            coverage_sufficient=(
                coverage_sufficient
            ),
            assessed_inspection_types=assessed,
            missing_inspection_types=missing,
            component_scores=components,
            critical_findings=critical_findings,
            recommendations=recommendations,
            requires_human_review=(
                any_findings
                or not coverage_sufficient
            ),
            safety_escalation_recommended=bool(
                critical_findings
            ),
            assessed_at=datetime.now(
                timezone.utc
            ),
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # DETECTOR RESULT INTEGRATION
    # ========================================================

    def calculate_from_detectors(
        self,
        detector_results: Mapping[
            str,
            Any,
        ],
        *,
        company_id: str,
        asset_id: str,
        equipment_id: str | None = None,
    ) -> VisualHealthScoreResult:
        inspections = [
            VisualInspectionAdapter.from_detector(
                kind,
                result,
            )
            for kind, result
            in detector_results.items()
        ]

        return self.calculate(
            inspections,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # ASYNC
    # ========================================================

    async def calculate_async(
        self,
        inspections: Sequence[
            VisualInspectionInput
        ],
        *,
        company_id: str,
        asset_id: str,
        equipment_id: str | None = None,
    ) -> VisualHealthScoreResult:
        return await asyncio.to_thread(
            self.calculate,
            inspections,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # BATCH
    # ========================================================

    def calculate_batch(
        self,
        assets: Sequence[
            Mapping[str, Any]
        ],
        *,
        company_id: str,
    ) -> list[VisualHealthScoreResult]:
        results = []

        for asset in assets:
            results.append(
                self.calculate(
                    asset["inspections"],
                    company_id=company_id,
                    asset_id=asset["asset_id"],
                    equipment_id=asset.get(
                        "equipment_id"
                    ),
                )
            )

        return results


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    engine = VisualHealthScoreEngine()

    inspections = [
        VisualInspectionInput(
            inspection_type="crack",
            detected=True,
            severity="moderate",
            finding_count=2,
        ),
        VisualInspectionInput(
            inspection_type="corrosion",
            detected=True,
            severity="low",
            affected_area_percent=3.5,
            finding_count=2,
        ),
        VisualInspectionInput(
            inspection_type="damage",
            detected=False,
            severity="none",
        ),
        VisualInspectionInput(
            inspection_type="leakage",
            detected=False,
            severity="none",
        ),
        VisualInspectionInput(
            inspection_type="overheating",
            detected=True,
            severity="high",
            affected_area_percent=4.0,
            finding_count=1,
        ),
    ]

    result = engine.calculate(
        inspections,
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="PUMP-101",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()
