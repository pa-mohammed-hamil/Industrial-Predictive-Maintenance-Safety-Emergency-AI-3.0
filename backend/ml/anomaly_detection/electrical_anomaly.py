# app/ml/anomaly_detection/electrical_anomaly.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


# ============================================================
# Enums
# ============================================================


class ElectricalAnomalyType(str, Enum):
    OVERVOLTAGE = "overvoltage"
    UNDERVOLTAGE = "undervoltage"
    OVERCURRENT = "overcurrent"
    UNDERCURRENT = "undercurrent"
    POWER_SURGE = "power_surge"
    POWER_DROP = "power_drop"
    FREQUENCY_HIGH = "frequency_high"
    FREQUENCY_LOW = "frequency_low"
    FREQUENCY_DEVIATION = "frequency_deviation"
    LOW_POWER_FACTOR = "low_power_factor"
    VOLTAGE_IMBALANCE = "voltage_imbalance"
    CURRENT_IMBALANCE = "current_imbalance"
    LEAKAGE_CURRENT = "leakage_current"
    RAPID_VOLTAGE_CHANGE = "rapid_voltage_change"
    RAPID_CURRENT_CHANGE = "rapid_current_change"
    PHASE_LOSS = "phase_loss"
    SENSOR_FAULT = "sensor_fault"
    MULTI_SIGNAL = "multi_signal"
    UNKNOWN = "unknown"


class ElectricalSeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ElectricalState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ANOMALOUS = "anomalous"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class Phase(str, Enum):
    A = "a"
    B = "b"
    C = "c"
    SINGLE = "single"
    UNKNOWN = "unknown"


# ============================================================
# Data Models
# ============================================================


@dataclass(slots=True)
class ElectricalReading:
    """
    Normalized electrical telemetry for one asset.

    All fields are optional because industrial telemetry can be
    incomplete or collected from different classes of equipment.
    """

    timestamp: datetime

    asset_id: str | None = None
    sensor_id: str | None = None
    site_id: str | None = None
    zone_id: str | None = None

    voltage_v: float | None = None
    current_a: float | None = None
    active_power_kw: float | None = None
    apparent_power_kva: float | None = None
    reactive_power_kvar: float | None = None

    frequency_hz: float | None = None
    power_factor: float | None = None
    leakage_current_ma: float | None = None

    phase_voltages_v: Mapping[str, float] | None = None
    phase_currents_a: Mapping[str, float] | None = None

    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

        for name in (
            "voltage_v",
            "current_a",
            "active_power_kw",
            "apparent_power_kva",
            "reactive_power_kvar",
            "frequency_hz",
            "power_factor",
            "leakage_current_ma",
        ):
            value = getattr(self, name)

            if value is not None and not _valid_number(value):
                raise ValueError(f"{name} must be a finite number")

        if self.power_factor is not None:
            self.power_factor = _clamp(self.power_factor, -1.0, 1.0)


@dataclass(slots=True)
class ElectricalThresholds:
    """
    Asset-specific electrical operating envelope.

    Defaults are intentionally generic. Production deployments
    should load these values from equipment specifications,
    commissioning data, or the central safety policy.
    """

    nominal_voltage_v: float = 230.0
    voltage_warning_ratio: float = 0.10
    voltage_critical_ratio: float = 0.20

    rated_current_a: float = 100.0
    current_warning_ratio: float = 1.20
    current_critical_ratio: float = 1.50

    nominal_frequency_hz: float = 50.0
    frequency_warning_deviation_hz: float = 1.0
    frequency_critical_deviation_hz: float = 2.0

    minimum_power_factor: float = 0.85
    critical_power_factor: float = 0.65

    leakage_warning_ma: float = 20.0
    leakage_critical_ma: float = 30.0

    voltage_imbalance_warning: float = 0.02
    voltage_imbalance_critical: float = 0.05

    current_imbalance_warning: float = 0.10
    current_imbalance_critical: float = 0.25

    rapid_voltage_change_ratio: float = 0.15
    rapid_current_change_ratio: float = 0.30

    phase_loss_ratio: float = 0.20

    minimum_confidence: float = 0.60


@dataclass(slots=True)
class ElectricalEvidence:
    metric: str
    value: float
    threshold: float | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ElectricalAnomaly:
    anomaly_id: str
    anomaly_type: ElectricalAnomalyType
    severity: ElectricalSeverity
    score: float
    confidence: float

    timestamp: datetime
    asset_id: str | None

    evidence: list[ElectricalEvidence] = field(default_factory=list)
    requires_investigation: bool = False
    requires_safety_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ElectricalAssessment:
    assessment_id: str
    timestamp: datetime
    asset_id: str | None

    state: ElectricalState
    severity: ElectricalSeverity
    anomaly_score: float
    confidence: float

    anomalies: list[ElectricalAnomaly]
    evidence: list[ElectricalEvidence]

    voltage_imbalance: float | None = None
    current_imbalance: float | None = None

    requires_investigation: bool = False
    requires_maintenance_review: bool = False
    requires_safety_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        return self.severity not in {
            ElectricalSeverity.NORMAL,
            ElectricalSeverity.LOW,
        }

    @property
    def is_critical(self) -> bool:
        return self.severity == ElectricalSeverity.CRITICAL


# ============================================================
# Electrical Anomaly Detector
# ============================================================


class ElectricalAnomalyDetector:
    """
    Deterministic electrical anomaly detector.

    Responsibilities
    ----------------
    - Voltage anomalies
    - Current anomalies
    - Frequency anomalies
    - Power-factor anomalies
    - Leakage-current anomalies
    - Three-phase imbalance
    - Phase loss
    - Rapid electrical changes
    - Multi-signal anomaly aggregation

    It does NOT authorize:
    - equipment shutdown
    - emergency isolation
    - breaker operation
    - human evacuation

    Those decisions belong to the central safety/decision engine.
    """

    _SEVERITY_RANK = {
        ElectricalSeverity.NORMAL: 0,
        ElectricalSeverity.LOW: 1,
        ElectricalSeverity.MEDIUM: 2,
        ElectricalSeverity.HIGH: 3,
        ElectricalSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        thresholds: ElectricalThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or ElectricalThresholds()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def assess(
        self,
        reading: ElectricalReading,
        *,
        previous: ElectricalReading | None = None,
    ) -> ElectricalAssessment:

        if reading.confidence < self.thresholds.minimum_confidence:
            return self._low_confidence_assessment(reading)

        anomalies: list[ElectricalAnomaly] = []

        anomalies.extend(self._check_voltage(reading))
        anomalies.extend(self._check_current(reading))
        anomalies.extend(self._check_frequency(reading))
        anomalies.extend(self._check_power_factor(reading))
        anomalies.extend(self._check_leakage(reading))
        anomalies.extend(self._check_phase_voltage_imbalance(reading))
        anomalies.extend(self._check_phase_current_imbalance(reading))
        anomalies.extend(self._check_phase_loss(reading))

        if previous is not None:
            anomalies.extend(
                self._check_rate_of_change(
                    current=reading,
                    previous=previous,
                )
            )

        anomalies = self._deduplicate(anomalies)

        severity = self._overall_severity(anomalies)
        score = self._overall_score(anomalies)

        voltage_imbalance = self._imbalance(
            reading.phase_voltages_v
        )

        current_imbalance = self._imbalance(
            reading.phase_currents_a
        )

        evidence = [
            item
            for anomaly in anomalies
            for item in anomaly.evidence
        ]

        requires_safety_review = any(
            anomaly.requires_safety_review
            for anomaly in anomalies
        )

        requires_investigation = bool(anomalies)

        requires_maintenance_review = (
            severity
            in {
                ElectricalSeverity.MEDIUM,
                ElectricalSeverity.HIGH,
                ElectricalSeverity.CRITICAL,
            }
        )

        state = self._state(severity)

        return ElectricalAssessment(
            assessment_id=str(uuid4()),
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            state=state,
            severity=severity,
            anomaly_score=score,
            confidence=reading.confidence,
            anomalies=anomalies,
            evidence=evidence,
            voltage_imbalance=voltage_imbalance,
            current_imbalance=current_imbalance,
            requires_investigation=requires_investigation,
            requires_maintenance_review=requires_maintenance_review,
            requires_safety_review=requires_safety_review,
            metadata={
                "detector": self.__class__.__name__,
                "sensor_id": reading.sensor_id,
                "site_id": reading.site_id,
                "zone_id": reading.zone_id,
            },
        )

    def detect(
        self,
        reading: ElectricalReading,
        *,
        previous: ElectricalReading | None = None,
    ) -> list[ElectricalAnomaly]:
        return self.assess(
            reading,
            previous=previous,
        ).anomalies

    def assess_sequence(
        self,
        readings: Sequence[ElectricalReading],
    ) -> list[ElectricalAssessment]:

        if not readings:
            return []

        ordered = sorted(
            readings,
            key=lambda item: item.timestamp,
        )

        results: list[ElectricalAssessment] = []

        previous: ElectricalReading | None = None

        for reading in ordered:
            results.append(
                self.assess(
                    reading,
                    previous=previous,
                )
            )
            previous = reading

        return results

    # --------------------------------------------------------
    # Voltage
    # --------------------------------------------------------

    def _check_voltage(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        value = reading.voltage_v

        if value is None:
            return []

        nominal = self.thresholds.nominal_voltage_v

        deviation = (value - nominal) / max(nominal, 1e-9)
        absolute_deviation = abs(deviation)

        if absolute_deviation < self.thresholds.voltage_warning_ratio:
            return []

        if deviation > 0:
            anomaly_type = ElectricalAnomalyType.OVERVOLTAGE
        else:
            anomaly_type = ElectricalAnomalyType.UNDERVOLTAGE

        if absolute_deviation >= self.thresholds.voltage_critical_ratio:
            severity = ElectricalSeverity.CRITICAL
            score = 1.0
        else:
            severity = ElectricalSeverity.HIGH
            score = _normalize_between(
                absolute_deviation,
                self.thresholds.voltage_warning_ratio,
                self.thresholds.voltage_critical_ratio,
            )

        return [
            self._create_anomaly(
                reading,
                anomaly_type,
                severity,
                score,
                [
                    ElectricalEvidence(
                        metric="voltage_v",
                        value=value,
                        threshold=nominal,
                        message=(
                            f"Voltage deviation is "
                            f"{absolute_deviation * 100:.2f}%"
                        ),
                    )
                ],
            )
        ]

    # --------------------------------------------------------
    # Current
    # --------------------------------------------------------

    def _check_current(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        current = reading.current_a

        if current is None:
            return []

        rated = self.thresholds.rated_current_a

        ratio = current / max(rated, 1e-9)

        if ratio < self.thresholds.current_warning_ratio:
            return []

        if ratio >= self.thresholds.current_critical_ratio:
            severity = ElectricalSeverity.CRITICAL
            score = 1.0
        else:
            severity = ElectricalSeverity.HIGH
            score = _normalize_between(
                ratio,
                self.thresholds.current_warning_ratio,
                self.thresholds.current_critical_ratio,
            )

        return [
            self._create_anomaly(
                reading,
                ElectricalAnomalyType.OVERCURRENT,
                severity,
                score,
                [
                    ElectricalEvidence(
                        metric="current_a",
                        value=current,
                        threshold=rated,
                        message=(
                            f"Current is {ratio:.2f}× rated current"
                        ),
                    )
                ],
            )
        ]

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    def _check_frequency(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        frequency = reading.frequency_hz

        if frequency is None:
            return []

        nominal = self.thresholds.nominal_frequency_hz
        deviation = frequency - nominal
        absolute = abs(deviation)

        if absolute < self.thresholds.frequency_warning_deviation_hz:
            return []

        anomaly_type = (
            ElectricalAnomalyType.FREQUENCY_HIGH
            if deviation > 0
            else ElectricalAnomalyType.FREQUENCY_LOW
        )

        if absolute >= self.thresholds.frequency_critical_deviation_hz:
            severity = ElectricalSeverity.CRITICAL
            score = 1.0
        else:
            severity = ElectricalSeverity.HIGH
            score = _normalize_between(
                absolute,
                self.thresholds.frequency_warning_deviation_hz,
                self.thresholds.frequency_critical_deviation_hz,
            )

        return [
            self._create_anomaly(
                reading,
                anomaly_type,
                severity,
                score,
                [
                    ElectricalEvidence(
                        metric="frequency_hz",
                        value=frequency,
                        threshold=nominal,
                        message=(
                            f"Frequency deviation: {deviation:+.2f} Hz"
                        ),
                    )
                ],
            )
        ]

    # --------------------------------------------------------
    # Power Factor
    # --------------------------------------------------------

    def _check_power_factor(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        pf = reading.power_factor

        if pf is None:
            return []

        absolute_pf = abs(pf)

        if absolute_pf >= self.thresholds.minimum_power_factor:
            return []

        if absolute_pf <= self.thresholds.critical_power_factor:
            severity = ElectricalSeverity.HIGH
            score = 0.90
        else:
            severity = ElectricalSeverity.MEDIUM
            score = _inverse_normalize(
                absolute_pf,
                self.thresholds.critical_power_factor,
                self.thresholds.minimum_power_factor,
            )

        return [
            self._create_anomaly(
                reading,
                ElectricalAnomalyType.LOW_POWER_FACTOR,
                severity,
                score,
                [
                    ElectricalEvidence(
                        metric="power_factor",
                        value=pf,
                        threshold=self.thresholds.minimum_power_factor,
                        message="Power factor below recommended limit",
                    )
                ],
                safety_review=False,
            )
        ]

    # --------------------------------------------------------
    # Leakage Current
    # --------------------------------------------------------

    def _check_leakage(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        leakage = reading.leakage_current_ma

        if leakage is None:
            return []

        if leakage < self.thresholds.leakage_warning_ma:
            return []

        if leakage >= self.thresholds.leakage_critical_ma:
            severity = ElectricalSeverity.CRITICAL
            score = 1.0
        else:
            severity = ElectricalSeverity.HIGH
            score = _normalize_between(
                leakage,
                self.thresholds.leakage_warning_ma,
                self.thresholds.leakage_critical_ma,
            )

        return [
            self._create_anomaly(
                reading,
                ElectricalAnomalyType.LEAKAGE_CURRENT,
                severity,
                score,
                [
                    ElectricalEvidence(
                        metric="leakage_current_ma",
                        value=leakage,
                        threshold=self.thresholds.leakage_warning_ma,
                        message="Abnormal leakage current detected",
                    )
                ],
                safety_review=True,
            )
        ]

    # --------------------------------------------------------
    # Phase Imbalance
    # --------------------------------------------------------

    def _check_phase_voltage_imbalance(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        imbalance = self._imbalance(reading.phase_voltages_v)

        if imbalance is None:
            return []

        if imbalance < self.thresholds.voltage_imbalance_warning:
            return []

        if imbalance >= self.thresholds.voltage_imbalance_critical:
            severity = ElectricalSeverity.CRITICAL
            score = 1.0
        else:
            severity = ElectricalSeverity.HIGH
            score = _normalize_between(
                imbalance,
                self.thresholds.voltage_imbalance_warning,
                self.thresholds.voltage_imbalance_critical,
            )

        return [
            self._create_anomaly(
                reading,
                ElectricalAnomalyType.VOLTAGE_IMBALANCE,
                severity,
                score,
                [
                    ElectricalEvidence(
                        metric="voltage_imbalance",
                        value=imbalance,
                        threshold=(
                            self.thresholds.voltage_imbalance_warning
                        ),
                        message=(
                            f"Phase voltage imbalance "
                            f"{imbalance * 100:.2f}%"
                        ),
                    )
                ],
            )
        ]

    def _check_phase_current_imbalance(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        imbalance = self._imbalance(reading.phase_currents_a)

        if imbalance is None:
            return []

        if imbalance < self.thresholds.current_imbalance_warning:
            return []

        if imbalance >= self.thresholds.current_imbalance_critical:
            severity = ElectricalSeverity.CRITICAL
            score = 1.0
        else:
            severity = ElectricalSeverity.HIGH
            score = _normalize_between(
                imbalance,
                self.thresholds.current_imbalance_warning,
                self.thresholds.current_imbalance_critical,
            )

        return [
            self._create_anomaly(
                reading,
                ElectricalAnomalyType.CURRENT_IMBALANCE,
                severity,
                score,
                [
                    ElectricalEvidence(
                        metric="current_imbalance",
                        value=imbalance,
                        threshold=(
                            self.thresholds.current_imbalance_warning
                        ),
                        message=(
                            f"Phase current imbalance "
                            f"{imbalance * 100:.2f}%"
                        ),
                    )
                ],
            )
        ]

    # --------------------------------------------------------
    # Phase Loss
    # --------------------------------------------------------

    def _check_phase_loss(
        self,
        reading: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        phases = reading.phase_voltages_v

        if not phases or len(phases) < 3:
            return []

        values = [
            float(value)
            for value in phases.values()
            if _valid_number(value)
        ]

        if len(values) < 3:
            return []

        average = mean(values)

        if average <= 0:
            return []

        minimum = min(values)

        if minimum >= average * self.thresholds.phase_loss_ratio:
            return []

        return [
            self._create_anomaly(
                reading,
                ElectricalAnomalyType.PHASE_LOSS,
                ElectricalSeverity.CRITICAL,
                1.0,
                [
                    ElectricalEvidence(
                        metric="minimum_phase_voltage",
                        value=minimum,
                        threshold=(
                            average * self.thresholds.phase_loss_ratio
                        ),
                        message="Possible phase loss detected",
                    )
                ],
                safety_review=True,
            )
        ]

    # --------------------------------------------------------
    # Rate-of-change detection
    # --------------------------------------------------------

    def _check_rate_of_change(
        self,
        *,
        current: ElectricalReading,
        previous: ElectricalReading,
    ) -> list[ElectricalAnomaly]:

        anomalies: list[ElectricalAnomaly] = []

        if (
            current.voltage_v is not None
            and previous.voltage_v is not None
            and previous.voltage_v != 0
        ):
            change = abs(
                current.voltage_v - previous.voltage_v
            ) / abs(previous.voltage_v)

            if change >= self.thresholds.rapid_voltage_change_ratio:
                anomalies.append(
                    self._create_anomaly(
                        current,
                        ElectricalAnomalyType.RAPID_VOLTAGE_CHANGE,
                        ElectricalSeverity.HIGH,
                        min(1.0, change),
                        [
                            ElectricalEvidence(
                                metric="voltage_change_ratio",
                                value=change,
                                threshold=(
                                    self.thresholds
                                    .rapid_voltage_change_ratio
                                ),
                                message=(
                                    f"Rapid voltage change "
                                    f"{change * 100:.2f}%"
                                ),
                            )
                        ],
                    )
                )

        if (
            current.current_a is not None
            and previous.current_a is not None
            and previous.current_a != 0
        ):
            change = abs(
                current.current_a - previous.current_a
            ) / abs(previous.current_a)

            if change >= self.thresholds.rapid_current_change_ratio:
                anomalies.append(
                    self._create_anomaly(
                        current,
                        ElectricalAnomalyType.RAPID_CURRENT_CHANGE,
                        ElectricalSeverity.HIGH,
                        min(1.0, change),
                        [
                            ElectricalEvidence(
                                metric="current_change_ratio",
                                value=change,
                                threshold=(
                                    self.thresholds
                                    .rapid_current_change_ratio
                                ),
                                message=(
                                    f"Rapid current change "
                                    f"{change * 100:.2f}%"
                                ),
                            )
                        ],
                    )
                )

        return anomalies

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _create_anomaly(
        self,
        reading: ElectricalReading,
        anomaly_type: ElectricalAnomalyType,
        severity: ElectricalSeverity,
        score: float,
        evidence: list[ElectricalEvidence],
        *,
        safety_review: bool | None = None,
    ) -> ElectricalAnomaly:

        if safety_review is None:
            safety_review = severity in {
                ElectricalSeverity.HIGH,
                ElectricalSeverity.CRITICAL,
            }

        return ElectricalAnomaly(
            anomaly_id=str(uuid4()),
            anomaly_type=anomaly_type,
            severity=severity,
            score=_clamp(score),
            confidence=reading.confidence,
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            evidence=evidence,
            requires_investigation=True,
            requires_safety_review=safety_review,
            metadata={
                "sensor_id": reading.sensor_id,
                "site_id": reading.site_id,
                "zone_id": reading.zone_id,
            },
        )

    @staticmethod
    def _imbalance(
        phases: Mapping[str, float] | None,
    ) -> float | None:

        if not phases or len(phases) < 2:
            return None

        values = [
            float(value)
            for value in phases.values()
            if _valid_number(value)
        ]

        if len(values) < 2:
            return None

        average = mean(values)

        if average == 0:
            return None

        maximum_deviation = max(
            abs(value - average)
            for value in values
        )

        return maximum_deviation / abs(average)

    def _overall_severity(
        self,
        anomalies: Sequence[ElectricalAnomaly],
    ) -> ElectricalSeverity:

        if not anomalies:
            return ElectricalSeverity.NORMAL

        return max(
            (anomaly.severity for anomaly in anomalies),
            key=lambda severity: self._SEVERITY_RANK[severity],
        )

    @staticmethod
    def _overall_score(
        anomalies: Sequence[ElectricalAnomaly],
    ) -> float:

        if not anomalies:
            return 0.0

        scores = sorted(
            (a.score * a.confidence for a in anomalies),
            reverse=True,
        )

        strongest = scores[0]

        corroboration_bonus = min(
            0.15,
            max(0, len(scores) - 1) * 0.04,
        )

        return _clamp(strongest + corroboration_bonus)

    @staticmethod
    def _state(
        severity: ElectricalSeverity,
    ) -> ElectricalState:

        if severity == ElectricalSeverity.NORMAL:
            return ElectricalState.NORMAL

        if severity == ElectricalSeverity.LOW:
            return ElectricalState.WARNING

        if severity in {
            ElectricalSeverity.MEDIUM,
            ElectricalSeverity.HIGH,
        }:
            return ElectricalState.ANOMALOUS

        return ElectricalState.CRITICAL

    @staticmethod
    def _deduplicate(
        anomalies: Iterable[ElectricalAnomaly],
    ) -> list[ElectricalAnomaly]:

        selected: dict[
            ElectricalAnomalyType,
            ElectricalAnomaly,
        ] = {}

        rank = {
            ElectricalSeverity.NORMAL: 0,
            ElectricalSeverity.LOW: 1,
            ElectricalSeverity.MEDIUM: 2,
            ElectricalSeverity.HIGH: 3,
            ElectricalSeverity.CRITICAL: 4,
        }

        for anomaly in anomalies:
            existing = selected.get(anomaly.anomaly_type)

            if existing is None:
                selected[anomaly.anomaly_type] = anomaly
                continue

            if rank[anomaly.severity] > rank[existing.severity]:
                selected[anomaly.anomaly_type] = anomaly

            elif (
                rank[anomaly.severity]
                == rank[existing.severity]
                and anomaly.score > existing.score
            ):
                selected[anomaly.anomaly_type] = anomaly

        return list(selected.values())

    def _low_confidence_assessment(
        self,
        reading: ElectricalReading,
    ) -> ElectricalAssessment:

        return ElectricalAssessment(
            assessment_id=str(uuid4()),
            timestamp=reading.timestamp,
            asset_id=reading.asset_id,
            state=ElectricalState.UNKNOWN,
            severity=ElectricalSeverity.LOW,
            anomaly_score=0.0,
            confidence=reading.confidence,
            anomalies=[],
            evidence=[
                ElectricalEvidence(
                    metric="sensor_confidence",
                    value=reading.confidence,
                    threshold=self.thresholds.minimum_confidence,
                    message="Electrical telemetry confidence too low",
                )
            ],
            requires_investigation=True,
            requires_maintenance_review=False,
            requires_safety_review=False,
            metadata={
                "reason": "low_sensor_confidence",
                "sensor_id": reading.sensor_id,
            },
        )


# ============================================================
# Statistical utilities
# ============================================================


def z_score_anomaly(
    value: float,
    history: Sequence[float],
    *,
    threshold: float = 3.0,
) -> tuple[bool, float]:
    """
    Lightweight statistical anomaly helper.

    Returns:
        (is_anomaly, absolute_z_score)
    """

    clean = [
        float(item)
        for item in history
        if _valid_number(item)
    ]

    if len(clean) < 3:
        return False, 0.0

    mu = mean(clean)
    sigma = pstdev(clean)

    if sigma <= 1e-12:
        return False, 0.0

    z = abs((float(value) - mu) / sigma)

    return z >= threshold, z


def moving_average(
    values: Sequence[float],
    *,
    window: int = 5,
) -> list[float]:

    if window <= 0:
        raise ValueError("window must be greater than zero")

    clean = [
        float(value)
        for value in values
        if _valid_number(value)
    ]

    output: list[float] = []

    for index in range(len(clean)):
        start = max(0, index - window + 1)
        output.append(mean(clean[start : index + 1]))

    return output


# ============================================================
# Convenience API
# ============================================================


def detect_electrical_anomaly(
    *,
    voltage_v: float | None = None,
    current_a: float | None = None,
    frequency_hz: float | None = None,
    power_factor: float | None = None,
    leakage_current_ma: float | None = None,
    phase_voltages_v: Mapping[str, float] | None = None,
    phase_currents_a: Mapping[str, float] | None = None,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    confidence: float = 1.0,
    timestamp: datetime | None = None,
    detector: ElectricalAnomalyDetector | None = None,
) -> ElectricalAssessment:

    reading = ElectricalReading(
        timestamp=timestamp or datetime.now(timezone.utc),
        asset_id=asset_id,
        sensor_id=sensor_id,
        voltage_v=voltage_v,
        current_a=current_a,
        frequency_hz=frequency_hz,
        power_factor=power_factor,
        leakage_current_ma=leakage_current_ma,
        phase_voltages_v=phase_voltages_v,
        phase_currents_a=phase_currents_a,
        confidence=confidence,
    )

    engine = detector or default_electrical_anomaly_detector

    return engine.assess(reading)


# ============================================================
# Internal utility functions
# ============================================================


def _valid_number(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalize_between(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    if maximum <= minimum:
        return 1.0

    return _clamp(
        (value - minimum)
        / (maximum - minimum)
    )


def _inverse_normalize(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    if maximum <= minimum:
        return 1.0

    return _clamp(
        (maximum - value)
        / (maximum - minimum)
    )


# ============================================================
# Default detector
# ============================================================


default_electrical_anomaly_detector = ElectricalAnomalyDetector()


__all__ = [
    "ElectricalAnomalyType",
    "ElectricalSeverity",
    "ElectricalState",
    "Phase",
    "ElectricalReading",
    "ElectricalThresholds",
    "ElectricalEvidence",
    "ElectricalAnomaly",
    "ElectricalAssessment",
    "ElectricalAnomalyDetector",
    "detect_electrical_anomaly",
    "z_score_anomaly",
    "moving_average",
    "default_electrical_anomaly_detector",
]