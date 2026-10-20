# app/integrations/mqtt/topics.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MQTTTopic(str, Enum):
    """Global MQTT topic patterns."""

    # Telemetry
    TELEMETRY_ALL = "industrial/+/+/telemetry/#"
    TELEMETRY_SITE = "industrial/{site_id}/+/telemetry/#"
    TELEMETRY_ASSET = "industrial/{site_id}/{asset_id}/telemetry/#"

    # Sensors
    SENSOR_ALL = "industrial/+/+/sensors/#"
    TEMPERATURE = "industrial/{site_id}/{asset_id}/sensors/temperature"
    VIBRATION = "industrial/{site_id}/{asset_id}/sensors/vibration"
    PRESSURE = "industrial/{site_id}/{asset_id}/sensors/pressure"
    CURRENT = "industrial/{site_id}/{asset_id}/sensors/current"
    VOLTAGE = "industrial/{site_id}/{asset_id}/sensors/voltage"
    ENERGY = "industrial/{site_id}/{asset_id}/sensors/energy"

    # Machine state
    MACHINE_STATUS = "industrial/{site_id}/{asset_id}/status"
    MACHINE_HEALTH = "industrial/{site_id}/{asset_id}/health"

    # Predictive maintenance
    PREDICTION = "industrial/{site_id}/{asset_id}/predictions"
    FAILURE_RISK = "industrial/{site_id}/{asset_id}/predictions/failure"
    RUL = "industrial/{site_id}/{asset_id}/predictions/rul"
    ANOMALY = "industrial/{site_id}/{asset_id}/anomalies"

    # Safety
    SAFETY_ALL = "industrial/+/+/safety/#"
    SAFETY_EVENT = "industrial/{site_id}/{asset_id}/safety/event"
    HAZARD = "industrial/{site_id}/{asset_id}/safety/hazard"

    FIRE = "industrial/{site_id}/{asset_id}/safety/fire"
    SMOKE = "industrial/{site_id}/{asset_id}/safety/smoke"
    OVERHEATING = "industrial/{site_id}/{asset_id}/safety/overheating"
    MOTION = "industrial/{site_id}/{asset_id}/safety/motion"
    PERSON = "industrial/{site_id}/{asset_id}/safety/person"
    RESTRICTED_ZONE = (
        "industrial/{site_id}/{asset_id}/safety/restricted-zone"
    )
    UNSAFE_PROXIMITY = (
        "industrial/{site_id}/{asset_id}/safety/unsafe-proximity"
    )

    # Incidents
    INCIDENT = "industrial/{site_id}/{asset_id}/incidents"
    INCIDENT_ALL = "industrial/+/+/incidents/#"

    # Emergency
    EMERGENCY = "industrial/{site_id}/{asset_id}/emergency"
    EMERGENCY_ALL = "industrial/+/+/emergency/#"

    # Maintenance
    MAINTENANCE = "industrial/{site_id}/{asset_id}/maintenance"
    WORK_ORDER = "industrial/{site_id}/{asset_id}/maintenance/work-order"

    # Cost intelligence
    COST = "industrial/{site_id}/{asset_id}/cost"
    DOWNTIME_COST = "industrial/{site_id}/{asset_id}/cost/downtime"
    REPAIR_COST = "industrial/{site_id}/{asset_id}/cost/repair"
    REPLACEMENT_COST = "industrial/{site_id}/{asset_id}/cost/replacement"

    # Device commands
    COMMAND = "industrial/{site_id}/{asset_id}/commands"
    COMMAND_ACK = "industrial/{site_id}/{asset_id}/commands/ack"

    # Heartbeats
    HEARTBEAT = "industrial/{site_id}/{asset_id}/heartbeat"


class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    VOLTAGE = "voltage"
    ENERGY = "energy"


class SafetyEventType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    MOTION = "motion"
    PERSON = "person"
    RESTRICTED_ZONE = "restricted-zone"
    UNSAFE_PROXIMITY = "unsafe-proximity"
    HAZARD = "hazard"


@dataclass(frozen=True, slots=True)
class TopicContext:
    site_id: str
    asset_id: str


def build_topic(
    topic: MQTTTopic | str,
    *,
    site_id: str,
    asset_id: str,
) -> str:
    """
    Build a concrete MQTT topic from a topic template.

    Example:
        build_topic(
            MQTTTopic.TEMPERATURE,
            site_id="factory-01",
            asset_id="motor-001",
        )

    Returns:
        industrial/factory-01/motor-001/sensors/temperature
    """

    if not site_id:
        raise ValueError("site_id cannot be empty")

    if not asset_id:
        raise ValueError("asset_id cannot be empty")

    value = topic.value if isinstance(topic, MQTTTopic) else topic

    return value.format(
        site_id=_sanitize_segment(site_id),
        asset_id=_sanitize_segment(asset_id),
    )


def sensor_topic(
    *,
    site_id: str,
    asset_id: str,
    sensor_type: SensorType | str,
) -> str:
    """Build a sensor telemetry topic."""

    sensor = (
        sensor_type.value
        if isinstance(sensor_type, SensorType)
        else sensor_type
    )

    return (
        f"industrial/"
        f"{_sanitize_segment(site_id)}/"
        f"{_sanitize_segment(asset_id)}/"
        f"sensors/"
        f"{_sanitize_segment(sensor)}"
    )


def safety_topic(
    *,
    site_id: str,
    asset_id: str,
    event_type: SafetyEventType | str,
) -> str:
    """Build a safety-event MQTT topic."""

    event = (
        event_type.value
        if isinstance(event_type, SafetyEventType)
        else event_type
    )

    return (
        f"industrial/"
        f"{_sanitize_segment(site_id)}/"
        f"{_sanitize_segment(asset_id)}/"
        f"safety/"
        f"{_sanitize_segment(event)}"
    )


def telemetry_topic(
    *,
    site_id: str,
    asset_id: str,
    metric: str | None = None,
) -> str:
    """
    Build a generic telemetry topic.

    Without metric:
        industrial/site/asset/telemetry

    With metric:
        industrial/site/asset/telemetry/rpm
    """

    base = (
        f"industrial/"
        f"{_sanitize_segment(site_id)}/"
        f"{_sanitize_segment(asset_id)}/"
        "telemetry"
    )

    if metric:
        return f"{base}/{_sanitize_segment(metric)}"

    return base


def command_topic(
    *,
    site_id: str,
    asset_id: str,
    command: str | None = None,
) -> str:
    """Build an equipment command topic."""

    base = build_topic(
        MQTTTopic.COMMAND,
        site_id=site_id,
        asset_id=asset_id,
    )

    if command:
        return f"{base}/{_sanitize_segment(command)}"

    return base


def site_wildcard(site_id: str) -> str:
    """Subscribe to every event from a site."""

    return f"industrial/{_sanitize_segment(site_id)}/#"


def asset_wildcard(
    site_id: str,
    asset_id: str,
) -> str:
    """Subscribe to every event for one asset."""

    return (
        f"industrial/"
        f"{_sanitize_segment(site_id)}/"
        f"{_sanitize_segment(asset_id)}/#"
    )


def sensor_wildcard(
    site_id: str,
    asset_id: str,
) -> str:
    """Subscribe to every sensor for one asset."""

    return (
        f"industrial/"
        f"{_sanitize_segment(site_id)}/"
        f"{_sanitize_segment(asset_id)}/"
        "sensors/#"
    )


def safety_wildcard(
    site_id: str,
    asset_id: str,
) -> str:
    """Subscribe to every safety event for one asset."""

    return (
        f"industrial/"
        f"{_sanitize_segment(site_id)}/"
        f"{_sanitize_segment(asset_id)}/"
        "safety/#"
    )


def parse_topic(topic: str) -> dict[str, str | None]:
    """
    Parse an industrial MQTT topic.

    Example input:
        industrial/factory-01/motor-001/sensors/temperature

    Returns:
        {
            "site_id": "factory-01",
            "asset_id": "motor-001",
            "category": "sensors",
            "event": "temperature",
        }
    """

    parts = topic.strip("/").split("/")

    if len(parts) < 4:
        raise ValueError(
            f"Invalid industrial MQTT topic: {topic}"
        )

    if parts[0] != "industrial":
        raise ValueError(
            f"Unsupported MQTT namespace: {parts[0]}"
        )

    return {
        "site_id": parts[1],
        "asset_id": parts[2],
        "category": parts[3],
        "event": "/".join(parts[4:]) if len(parts) > 4 else None,
    }


def _sanitize_segment(value: str) -> str:
    """
    Validate a single MQTT topic segment.

    MQTT wildcard characters are intentionally rejected here because
    these helpers generate concrete publish topics.
    """

    value = value.strip()

    if not value:
        raise ValueError("MQTT topic segment cannot be empty")

    if "/" in value:
        raise ValueError(
            f"MQTT topic segment cannot contain '/': {value!r}"
        )

    if "+" in value or "#" in value:
        raise ValueError(
            f"MQTT wildcards are not allowed in topic segment: {value!r}"
        )

    return value