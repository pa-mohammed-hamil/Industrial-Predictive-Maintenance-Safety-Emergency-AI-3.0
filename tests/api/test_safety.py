
"""
INDUSTRA AI
tests/test_safety.py

Industrial Safety & Emergency Intelligence Tests

Run:
    pip install pytest
    pytest tests/test_safety.py -v

This file uses a reference safety engine for standalone testing.
Replace SafetyEngine with your actual production safety service
to validate application behavior.

IMPORTANT:
No physical equipment, relays, emergency-stop circuits,
or industrial controllers are operated by these tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

import pytest


# ============================================================
# ENUMS
# ============================================================

class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class EmergencyType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    GAS_LEAK = "gas_leak"
    WORKER_INJURY = "worker_injury"
    EQUIPMENT_FAILURE = "equipment_failure"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class SensorReading:
    asset_id: str
    temperature: float
    vibration: float
    pressure: float
    gas_ppm: float = 0.0
    smoke_detected: bool = False
    flame_detected: bool = False


@dataclass
class WorkerObservation:
    worker_id: str
    helmet: bool
    vest: bool
    gloves: bool
    restricted_zone: bool = False


@dataclass
class SafetyAlert:
    alert_id: str
    asset_id: str
    alert_type: str
    severity: Severity
    message: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class Incident:
    incident_id: str
    alert_id: str
    status: IncidentStatus = IncidentStatus.OPEN
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None


# ============================================================
# REFERENCE SAFETY ENGINE
# ============================================================

class SafetyEngine:
    """
    Isolated reference implementation.

    Thresholds are illustrative test values only.
    They are NOT certified industrial safety limits.
    """

    MAX_TEMPERATURE = 85.0
    MAX_VIBRATION = 5.5
    MAX_PRESSURE = 9.0
    MAX_GAS_PPM = 100.0

    def __init__(self):
        self.alerts: list[SafetyAlert] = []
        self.incidents: dict[str, Incident] = {}
        self.audit_logs: list[dict] = []

    def validate_sensor(self, reading: SensorReading) -> bool:
        if not reading.asset_id.strip():
            raise ValueError("Asset ID is required")

        values = (
            reading.temperature,
            reading.vibration,
            reading.pressure,
            reading.gas_ppm,
        )

        import math

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Sensor values must be finite")

        if reading.vibration < 0:
            raise ValueError("Vibration cannot be negative")

        if reading.pressure < 0:
            raise ValueError("Pressure cannot be negative")

        if reading.gas_ppm < 0:
            raise ValueError("Gas concentration cannot be negative")

        return True

    def create_alert(
        self,
        asset_id: str,
        alert_type: str,
        severity: Severity,
        message: str,
    ) -> SafetyAlert:

        alert = SafetyAlert(
            alert_id=str(uuid4()),
            asset_id=asset_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
        )

        self.alerts.append(alert)

        self.audit_logs.append({
            "action": "alert_created",
            "alert_id": alert.alert_id,
            "severity": severity.value,
        })

        return alert

    def evaluate_sensor(
        self,
        reading: SensorReading,
    ) -> list[SafetyAlert]:

        self.validate_sensor(reading)

        alerts = []

        if reading.flame_detected:
            alerts.append(self.create_alert(
                reading.asset_id,
                "fire",
                Severity.CRITICAL,
                "Flame detected",
            ))

        if reading.smoke_detected:
            alerts.append(self.create_alert(
                reading.asset_id,
                "smoke",
                Severity.CRITICAL,
                "Smoke detected",
            ))

        if reading.temperature > self.MAX_TEMPERATURE:
            alerts.append(self.create_alert(
                reading.asset_id,
                "overheating",
                Severity.HIGH,
                "Temperature exceeds demo threshold",
            ))

        if reading.vibration > self.MAX_VIBRATION:
            alerts.append(self.create_alert(
                reading.asset_id,
                "high_vibration",
                Severity.HIGH,
                "Vibration exceeds demo threshold",
            ))

        if reading.pressure > self.MAX_PRESSURE:
            alerts.append(self.create_alert(
                reading.asset_id,
                "high_pressure",
                Severity.HIGH,
                "Pressure exceeds demo threshold",
            ))

        if reading.gas_ppm > self.MAX_GAS_PPM:
            alerts.append(self.create_alert(
                reading.asset_id,
                "gas_leak",
                Severity.CRITICAL,
                "Gas concentration exceeds demo threshold",
            ))

        return alerts

    def evaluate_worker(
        self,
        observation: WorkerObservation,
    ) -> list[SafetyAlert]:

        if not observation.worker_id.strip():
            raise ValueError("Worker ID is required")

        alerts = []

        if not observation.helmet:
            alerts.append(self.create_alert(
                observation.worker_id,
                "missing_helmet",
                Severity.HIGH,
                "Worker helmet not detected",
            ))

        if not observation.vest:
            alerts.append(self.create_alert(
                observation.worker_id,
                "missing_vest",
                Severity.MEDIUM,
                "Worker safety vest not detected",
            ))

        if not observation.gloves:
            alerts.append(self.create_alert(
                observation.worker_id,
                "missing_gloves",
                Severity.MEDIUM,
                "Worker gloves not detected",
            ))

        if observation.restricted_zone:
            alerts.append(self.create_alert(
                observation.worker_id,
                "restricted_zone",
                Severity.CRITICAL,
                "Worker detected in restricted zone",
            ))

        return alerts

    def create_incident(self, alert_id: str) -> Incident:
        if not any(a.alert_id == alert_id for a in self.alerts):
            raise ValueError("Unknown alert")

        incident = Incident(
            incident_id=str(uuid4()),
            alert_id=alert_id,
        )

        self.incidents[incident.incident_id] = incident

        self.audit_logs.append({
            "action": "incident_created",
            "incident_id": incident.incident_id,
        })

        return incident

    def acknowledge_incident(
        self,
        incident_id: str,
        user: str,
    ) -> Incident:

        if not user.strip():
            raise PermissionError("Authenticated user required")

        incident = self.incidents[incident_id]

        if incident.status != IncidentStatus.OPEN:
            raise ValueError("Incident is not open")

        incident.status = IncidentStatus.ACKNOWLEDGED
        incident.acknowledged_by = user

        self.audit_logs.append({
            "action": "incident_acknowledged",
            "incident_id": incident_id,
            "user": user,
        })

        return incident

    def resolve_incident(
        self,
        incident_id: str,
        user: str,
    ) -> Incident:

        if not user.strip():
            raise PermissionError("Authenticated user required")

        incident = self.incidents[incident_id]

        if incident.status != IncidentStatus.ACKNOWLEDGED:
            raise ValueError("Incident must be acknowledged first")

        incident.status = IncidentStatus.RESOLVED
        incident.resolved_by = user

        self.audit_logs.append({
            "action": "incident_resolved",
            "incident_id": incident_id,
            "user": user,
        })

        return incident

    def emergency_escalation(
        self,
        alert: SafetyAlert,
    ) -> bool:

        return alert.severity == Severity.CRITICAL

    def authorize_shutdown(
        self,
        role: str,
        confirmed: bool,
    ) -> bool:
        """
        Simulated authorization decision only.

        This method never controls physical machinery.
        """
        allowed_roles = {
            "safety_officer",
            "plant_manager",
        }

        return role in allowed_roles and confirmed


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def engine():
    return SafetyEngine()


@pytest.fixture
def normal_sensor():
    return SensorReading(
        asset_id="MTR-001",
        temperature=65,
        vibration=3.2,
        pressure=5.0,
        gas_ppm=10,
    )


@pytest.fixture
def critical_sensor():
    return SensorReading(
        asset_id="GEN-001",
        temperature=98,
        vibration=7.5,
        pressure=11.0,
        gas_ppm=150,
        smoke_detected=True,
        flame_detected=True,
    )


@pytest.fixture
def compliant_worker():
    return WorkerObservation(
        worker_id="WRK-001",
        helmet=True,
        vest=True,
        gloves=True,
    )


# ============================================================
# SENSOR VALIDATION TESTS
# ============================================================

def test_valid_sensor_reading(engine, normal_sensor):
    assert engine.validate_sensor(normal_sensor)


def test_missing_asset_id(engine, normal_sensor):
    normal_sensor.asset_id = ""

    with pytest.raises(ValueError):
        engine.validate_sensor(normal_sensor)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vibration", -1),
        ("pressure", -2),
        ("gas_ppm", -10),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("vibration", float("nan")),
    ],
)
def test_invalid_sensor_values(
    engine, normal_sensor, field, value
):
    setattr(normal_sensor, field, value)

    with pytest.raises(ValueError):
        engine.validate_sensor(normal_sensor)


# ============================================================
# FIRE AND SMOKE DETECTION TESTS
# ============================================================

def test_fire_detection(engine, normal_sensor):
    normal_sensor.flame_detected = True

    alerts = engine.evaluate_sensor(normal_sensor)

    assert any(a.alert_type == "fire" for a in alerts)


def test_fire_severity_is_critical(engine, normal_sensor):
    normal_sensor.flame_detected = True

    alerts = engine.evaluate_sensor(normal_sensor)

    fire = next(a for a in alerts if a.alert_type == "fire")

    assert fire.severity == Severity.CRITICAL


def test_smoke_detection(engine, normal_sensor):
    normal_sensor.smoke_detected = True

    alerts = engine.evaluate_sensor(normal_sensor)

    assert any(a.alert_type == "smoke" for a in alerts)


def test_fire_and_smoke_generate_separate_alerts(
    engine, normal_sensor
):
    normal_sensor.flame_detected = True
    normal_sensor.smoke_detected = True

    alerts = engine.evaluate_sensor(normal_sensor)

    assert len(alerts) == 2


# ============================================================
# OVERHEATING TESTS
# ============================================================

def test_overheating_detection(engine, normal_sensor):
    normal_sensor.temperature = 90

    alerts = engine.evaluate_sensor(normal_sensor)

    assert any(
        a.alert_type == "overheating" for a in alerts
    )


def test_temperature_at_threshold(engine, normal_sensor):
    normal_sensor.temperature = 85

    alerts = engine.evaluate_sensor(normal_sensor)

    assert not any(
        a.alert_type == "overheating" for a in alerts
    )


def test_temperature_above_threshold(engine, normal_sensor):
    normal_sensor.temperature = 85.1

    alerts = engine.evaluate_sensor(normal_sensor)

    assert any(
        a.alert_type == "overheating" for a in alerts
    )


# ============================================================
# VIBRATION TESTS
# ============================================================

def test_high_vibration(engine, normal_sensor):
    normal_sensor.vibration = 7.0

    alerts = engine.evaluate_sensor(normal_sensor)

    assert any(
        a.alert_type == "high_vibration" for a in alerts
    )


def test_vibration_at_threshold(engine, normal_sensor):
    normal_sensor.vibration = 5.5

    alerts = engine.evaluate_sensor(normal_sensor)

    assert not any(
        a.alert_type == "high_vibration" for a in alerts
    )


# ============================================================
# PRESSURE TESTS
# ============================================================

def test_high_pressure(engine, normal_sensor):
    normal_sensor.pressure = 11

    alerts = engine.evaluate_sensor(normal_sensor)

    assert any(
        a.alert_type == "high_pressure" for a in alerts
    )


def test_pressure_at_threshold(engine, normal_sensor):
    normal_sensor.pressure = 9

    alerts = engine.evaluate_sensor(normal_sensor)

    assert not any(
        a.alert_type == "high_pressure" for a in alerts
    )


# ============================================================
# GAS LEAK TESTS
# ============================================================

def test_gas_leak_detection(engine, normal_sensor):
    normal_sensor.gas_ppm = 150

    alerts = engine.evaluate_sensor(normal_sensor)

    assert any(a.alert_type == "gas_leak" for a in alerts)


def test_gas_leak_is_critical(engine, normal_sensor):
    normal_sensor.gas_ppm = 150

    alerts = engine.evaluate_sensor(normal_sensor)

    gas_alert = next(
        a for a in alerts if a.alert_type == "gas_leak"
    )

    assert gas_alert.severity == Severity.CRITICAL


def test_gas_at_threshold(engine, normal_sensor):
    normal_sensor.gas_ppm = 100

    alerts = engine.evaluate_sensor(normal_sensor)

    assert not any(a.alert_type == "gas_leak" for a in alerts)


# ============================================================
# MULTIPLE HAZARD TESTS
# ============================================================

def test_multiple_hazards(engine, critical_sensor):
    alerts = engine.evaluate_sensor(critical_sensor)

    types = {a.alert_type for a in alerts}

    assert types == {
        "fire",
        "smoke",
        "overheating",
        "high_vibration",
        "high_pressure",
        "gas_leak",
    }


def test_normal_sensor_no_alerts(engine, normal_sensor):
    alerts = engine.evaluate_sensor(normal_sensor)

    assert alerts == []


# ============================================================
# WORKER PPE TESTS
# ============================================================

def test_worker_ppe_compliant(engine, compliant_worker):
    alerts = engine.evaluate_worker(compliant_worker)

    assert alerts == []


def test_missing_helmet(engine, compliant_worker):
    compliant_worker.helmet = False

    alerts = engine.evaluate_worker(compliant_worker)

    assert any(
        a.alert_type == "missing_helmet" for a in alerts
    )


def test_missing_safety_vest(engine, compliant_worker):
    compliant_worker.vest = False

    alerts = engine.evaluate_worker(compliant_worker)

    assert any(
        a.alert_type == "missing_vest" for a in alerts
    )


def test_missing_gloves(engine, compliant_worker):
    compliant_worker.gloves = False

    alerts = engine.evaluate_worker(compliant_worker)

    assert any(
        a.alert_type == "missing_gloves" for a in alerts
    )


def test_multiple_ppe_violations(engine, compliant_worker):
    compliant_worker.helmet = False
    compliant_worker.vest = False
    compliant_worker.gloves = False

    alerts = engine.evaluate_worker(compliant_worker)

    assert len(alerts) == 3


def test_worker_missing_id(engine, compliant_worker):
    compliant_worker.worker_id = ""

    with pytest.raises(ValueError):
        engine.evaluate_worker(compliant_worker)


# ============================================================
# RESTRICTED ZONE TESTS
# ============================================================

def test_restricted_zone_violation(engine, compliant_worker):
    compliant_worker.restricted_zone = True

    alerts = engine.evaluate_worker(compliant_worker)

    assert any(
        a.alert_type == "restricted_zone" for a in alerts
    )


def test_restricted_zone_is_critical(engine, compliant_worker):
    compliant_worker.restricted_zone = True

    alerts = engine.evaluate_worker(compliant_worker)

    alert = next(
        a for a in alerts
        if a.alert_type == "restricted_zone"
    )

    assert alert.severity == Severity.CRITICAL


# ============================================================
# ALERT MANAGEMENT TESTS
# ============================================================

def test_alert_creation(engine):
    alert = engine.create_alert(
        "MTR-001",
        "overheating",
        Severity.HIGH,
        "Temperature warning",
    )

    assert alert.alert_id
    assert alert.asset_id == "MTR-001"
    assert alert.severity == Severity.HIGH


def test_alert_id_unique(engine):
    first = engine.create_alert(
        "MTR-001", "fire", Severity.CRITICAL, "Fire"
    )

    second = engine.create_alert(
        "MTR-001", "fire", Severity.CRITICAL, "Fire"
    )

    assert first.alert_id != second.alert_id


def test_alert_timestamp_timezone_aware(engine):
    alert = engine.create_alert(
        "MTR-001", "smoke", Severity.CRITICAL, "Smoke"
    )

    assert alert.timestamp.tzinfo is not None


def test_alert_persisted_in_memory(engine):
    alert = engine.create_alert(
        "MTR-001", "fire", Severity.CRITICAL, "Fire"
    )

    assert alert in engine.alerts


# ============================================================
# EMERGENCY ESCALATION TESTS
# ============================================================

def test_critical_alert_escalates(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    assert engine.emergency_escalation(alert)


def test_high_alert_does_not_auto_escalate(engine):
    alert = engine.create_alert(
        "MTR-001", "overheating",
        Severity.HIGH, "Overheating"
    )

    assert not engine.emergency_escalation(alert)


@pytest.mark.parametrize(
    "severity,expected",
    [
        (Severity.INFO, False),
        (Severity.LOW, False),
        (Severity.MEDIUM, False),
        (Severity.HIGH, False),
        (Severity.CRITICAL, True),
    ],
)
def test_escalation_severity_mapping(
    engine, severity, expected
):
    alert = engine.create_alert(
        "MTR-001", "test", severity, "Test alert"
    )

    assert engine.emergency_escalation(alert) is expected


# ============================================================
# INCIDENT MANAGEMENT TESTS
# ============================================================

def test_create_incident(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    incident = engine.create_incident(alert.alert_id)

    assert incident.status == IncidentStatus.OPEN


def test_create_incident_unknown_alert(engine):
    with pytest.raises(ValueError):
        engine.create_incident("unknown-alert")


def test_acknowledge_incident(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    incident = engine.create_incident(alert.alert_id)

    updated = engine.acknowledge_incident(
        incident.incident_id, "safety_officer"
    )

    assert updated.status == IncidentStatus.ACKNOWLEDGED
    assert updated.acknowledged_by == "safety_officer"


def test_acknowledge_requires_user(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    incident = engine.create_incident(alert.alert_id)

    with pytest.raises(PermissionError):
        engine.acknowledge_incident(incident.incident_id, "")


def test_resolve_incident(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    incident = engine.create_incident(alert.alert_id)

    engine.acknowledge_incident(
        incident.incident_id, "safety_officer"
    )

    resolved = engine.resolve_incident(
        incident.incident_id, "plant_manager"
    )

    assert resolved.status == IncidentStatus.RESOLVED


def test_cannot_resolve_unacknowledged_incident(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    incident = engine.create_incident(alert.alert_id)

    with pytest.raises(ValueError):
        engine.resolve_incident(
            incident.incident_id, "plant_manager"
        )


def test_cannot_acknowledge_twice(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    incident = engine.create_incident(alert.alert_id)

    engine.acknowledge_incident(
        incident.incident_id, "safety_officer"
    )

    with pytest.raises(ValueError):
        engine.acknowledge_incident(
            incident.incident_id, "plant_manager"
        )


# ============================================================
# EMERGENCY SHUTDOWN AUTHORIZATION TESTS
# ============================================================

@pytest.mark.parametrize(
    "role,confirmed,expected",
    [
        ("safety_officer", True, True),
        ("plant_manager", True, True),
        ("operator", True, False),
        ("viewer", True, False),
        ("guest", True, False),
        ("safety_officer", False, False),
        ("plant_manager", False, False),
    ],
)
def test_shutdown_authorization(
    engine, role, confirmed, expected
):
    assert engine.authorize_shutdown(
        role, confirmed
    ) is expected


def test_shutdown_never_automatically_executed(engine):
    """
    Authorization result must not represent
    physical shutdown execution.
    """
    authorized = engine.authorize_shutdown(
        "safety_officer", True
    )

    assert authorized is True
    assert not hasattr(engine, "execute_physical_shutdown")


# ============================================================
# AUDIT LOG TESTS
# ============================================================

def test_alert_creates_audit_log(engine):
    engine.create_alert(
        "MTR-001", "fire", Severity.CRITICAL, "Fire"
    )

    assert any(
        entry["action"] == "alert_created"
        for entry in engine.audit_logs
    )


def test_incident_lifecycle_audited(engine):
    alert = engine.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    incident = engine.create_incident(alert.alert_id)

    engine.acknowledge_incident(
        incident.incident_id, "safety_officer"
    )

    engine.resolve_incident(
        incident.incident_id, "plant_manager"
    )

    actions = [
        entry["action"]
        for entry in engine.audit_logs
    ]

    assert "alert_created" in actions
    assert "incident_created" in actions
    assert "incident_acknowledged" in actions
    assert "incident_resolved" in actions


# ============================================================
# DATA ISOLATION TESTS
# ============================================================

def test_engine_instances_have_separate_alerts():
    first = SafetyEngine()
    second = SafetyEngine()

    first.create_alert(
        "MTR-001", "fire", Severity.CRITICAL, "Fire"
    )

    assert len(first.alerts) == 1
    assert len(second.alerts) == 0


def test_engine_instances_have_separate_incidents():
    first = SafetyEngine()
    second = SafetyEngine()

    alert = first.create_alert(
        "GEN-001", "fire", Severity.CRITICAL, "Fire"
    )

    first.create_incident(alert.alert_id)

    assert len(first.incidents) == 1
    assert len(second.incidents) == 0


# ============================================================
# END-TO-END REFERENCE WORKFLOW TEST
# ============================================================

def test_fire_to_incident_resolution_workflow(engine):
    reading = SensorReading(
        asset_id="GEN-001",
        temperature=95,
        vibration=6.5,
        pressure=10,
        gas_ppm=120,
        smoke_detected=True,
        flame_detected=True,
    )

    alerts = engine.evaluate_sensor(reading)

    fire_alert = next(
        a for a in alerts
        if a.alert_type == "fire"
    )

    assert engine.emergency_escalation(fire_alert)

    incident = engine.create_incident(
        fire_alert.alert_id
    )

    assert incident.status == IncidentStatus.OPEN

    engine.acknowledge_incident(
        incident.incident_id,
        "safety_officer",
    )

    assert incident.status == IncidentStatus.ACKNOWLEDGED

    engine.resolve_incident(
        incident.incident_id,
        "plant_manager",
    )

    assert incident.status == IncidentStatus.RESOLVED

    assert len(engine.audit_logs) >= 4
