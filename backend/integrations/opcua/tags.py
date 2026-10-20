# app/integrations/opcua/tags.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class TagDataType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATETIME = "datetime"


class TagCategory(str, Enum):
    MACHINE = "machine"
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    ELECTRICAL = "electrical"
    ENERGY = "energy"
    PRODUCTION = "production"
    SAFETY = "safety"
    MAINTENANCE = "maintenance"
    HEALTH = "health"
    COMMAND = "command"


class TagAccess(str, Enum):
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True, slots=True)
class OPCUATag:
    name: str
    node_id: str

    category: TagCategory
    data_type: TagDataType

    unit: str | None = None
    description: str | None = None

    access: TagAccess = TagAccess.READ_ONLY

    minimum: float | None = None
    maximum: float | None = None

    warning_low: float | None = None
    warning_high: float | None = None

    critical_low: float | None = None
    critical_high: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def readable(self) -> bool:
        return self.access in {
            TagAccess.READ_ONLY,
            TagAccess.READ_WRITE,
        }

    @property
    def writable(self) -> bool:
        return self.access in {
            TagAccess.WRITE_ONLY,
            TagAccess.READ_WRITE,
        }


# ---------------------------------------------------------
# Machine
# ---------------------------------------------------------

MACHINE_RUNNING = OPCUATag(
    name="machine_running",
    node_id="ns=2;s=Machine.Status.Running",
    category=TagCategory.MACHINE,
    data_type=TagDataType.BOOLEAN,
    description="Machine running state",
)

MACHINE_SPEED = OPCUATag(
    name="machine_speed",
    node_id="ns=2;s=Machine.Speed",
    category=TagCategory.MACHINE,
    data_type=TagDataType.FLOAT,
    unit="rpm",
    minimum=0.0,
    maximum=10000.0,
    warning_high=8500.0,
    critical_high=9500.0,
)

MACHINE_LOAD = OPCUATag(
    name="machine_load",
    node_id="ns=2;s=Machine.Load",
    category=TagCategory.MACHINE,
    data_type=TagDataType.FLOAT,
    unit="%",
    minimum=0.0,
    maximum=100.0,
    warning_high=85.0,
    critical_high=95.0,
)

OPERATING_HOURS = OPCUATag(
    name="operating_hours",
    node_id="ns=2;s=Machine.OperatingHours",
    category=TagCategory.MACHINE,
    data_type=TagDataType.FLOAT,
    unit="hours",
)


# ---------------------------------------------------------
# Temperature
# ---------------------------------------------------------

BEARING_TEMPERATURE = OPCUATag(
    name="bearing_temperature",
    node_id="ns=2;s=Sensors.Temperature.Bearing",
    category=TagCategory.TEMPERATURE,
    data_type=TagDataType.FLOAT,
    unit="°C",
    minimum=-40.0,
    maximum=200.0,
    warning_high=80.0,
    critical_high=95.0,
)

MOTOR_TEMPERATURE = OPCUATag(
    name="motor_temperature",
    node_id="ns=2;s=Sensors.Temperature.Motor",
    category=TagCategory.TEMPERATURE,
    data_type=TagDataType.FLOAT,
    unit="°C",
    warning_high=90.0,
    critical_high=110.0,
)

AMBIENT_TEMPERATURE = OPCUATag(
    name="ambient_temperature",
    node_id="ns=2;s=Sensors.Temperature.Ambient",
    category=TagCategory.TEMPERATURE,
    data_type=TagDataType.FLOAT,
    unit="°C",
)


# ---------------------------------------------------------
# Vibration
# ---------------------------------------------------------

VIBRATION_RMS = OPCUATag(
    name="vibration_rms",
    node_id="ns=2;s=Sensors.Vibration.RMS",
    category=TagCategory.VIBRATION,
    data_type=TagDataType.FLOAT,
    unit="mm/s",
    minimum=0.0,
    warning_high=4.5,
    critical_high=7.1,
)

VIBRATION_X = OPCUATag(
    name="vibration_x",
    node_id="ns=2;s=Sensors.Vibration.X",
    category=TagCategory.VIBRATION,
    data_type=TagDataType.FLOAT,
    unit="mm/s",
)

VIBRATION_Y = OPCUATag(
    name="vibration_y",
    node_id="ns=2;s=Sensors.Vibration.Y",
    category=TagCategory.VIBRATION,
    data_type=TagDataType.FLOAT,
    unit="mm/s",
)

VIBRATION_Z = OPCUATag(
    name="vibration_z",
    node_id="ns=2;s=Sensors.Vibration.Z",
    category=TagCategory.VIBRATION,
    data_type=TagDataType.FLOAT,
    unit="mm/s",
)


# ---------------------------------------------------------
# Pressure
# ---------------------------------------------------------

INLET_PRESSURE = OPCUATag(
    name="inlet_pressure",
    node_id="ns=2;s=Sensors.Pressure.Inlet",
    category=TagCategory.PRESSURE,
    data_type=TagDataType.FLOAT,
    unit="bar",
)

OUTLET_PRESSURE = OPCUATag(
    name="outlet_pressure",
    node_id="ns=2;s=Sensors.Pressure.Outlet",
    category=TagCategory.PRESSURE,
    data_type=TagDataType.FLOAT,
    unit="bar",
)

OIL_PRESSURE = OPCUATag(
    name="oil_pressure",
    node_id="ns=2;s=Sensors.Pressure.Oil",
    category=TagCategory.PRESSURE,
    data_type=TagDataType.FLOAT,
    unit="bar",
    warning_low=1.5,
    critical_low=1.0,
)


# ---------------------------------------------------------
# Electrical
# ---------------------------------------------------------

CURRENT = OPCUATag(
    name="current",
    node_id="ns=2;s=Electrical.Current",
    category=TagCategory.ELECTRICAL,
    data_type=TagDataType.FLOAT,
    unit="A",
)

VOLTAGE = OPCUATag(
    name="voltage",
    node_id="ns=2;s=Electrical.Voltage",
    category=TagCategory.ELECTRICAL,
    data_type=TagDataType.FLOAT,
    unit="V",
)

POWER = OPCUATag(
    name="power",
    node_id="ns=2;s=Electrical.Power",
    category=TagCategory.ELECTRICAL,
    data_type=TagDataType.FLOAT,
    unit="kW",
)

POWER_FACTOR = OPCUATag(
    name="power_factor",
    node_id="ns=2;s=Electrical.PowerFactor",
    category=TagCategory.ELECTRICAL,
    data_type=TagDataType.FLOAT,
    minimum=0.0,
    maximum=1.0,
    warning_low=0.85,
)


# ---------------------------------------------------------
# Energy
# ---------------------------------------------------------

ENERGY_CONSUMPTION = OPCUATag(
    name="energy_consumption",
    node_id="ns=2;s=Energy.Total",
    category=TagCategory.ENERGY,
    data_type=TagDataType.FLOAT,
    unit="kWh",
)

ENERGY_RATE = OPCUATag(
    name="energy_rate",
    node_id="ns=2;s=Energy.Rate",
    category=TagCategory.ENERGY,
    data_type=TagDataType.FLOAT,
    unit="kW",
)


# ---------------------------------------------------------
# Production
# ---------------------------------------------------------

PRODUCTION_COUNT = OPCUATag(
    name="production_count",
    node_id="ns=2;s=Production.Count",
    category=TagCategory.PRODUCTION,
    data_type=TagDataType.INTEGER,
    unit="units",
)

PRODUCTION_RATE = OPCUATag(
    name="production_rate",
    node_id="ns=2;s=Production.Rate",
    category=TagCategory.PRODUCTION,
    data_type=TagDataType.FLOAT,
    unit="units/hour",
)

SCRAP_COUNT = OPCUATag(
    name="scrap_count",
    node_id="ns=2;s=Production.ScrapCount",
    category=TagCategory.PRODUCTION,
    data_type=TagDataType.INTEGER,
    unit="units",
)


# ---------------------------------------------------------
# Safety
# ---------------------------------------------------------

FIRE_DETECTED = OPCUATag(
    name="fire_detected",
    node_id="ns=2;s=Safety.FireDetected",
    category=TagCategory.SAFETY,
    data_type=TagDataType.BOOLEAN,
)

SMOKE_DETECTED = OPCUATag(
    name="smoke_detected",
    node_id="ns=2;s=Safety.SmokeDetected",
    category=TagCategory.SAFETY,
    data_type=TagDataType.BOOLEAN,
)

OVERHEATING = OPCUATag(
    name="overheating",
    node_id="ns=2;s=Safety.Overheating",
    category=TagCategory.SAFETY,
    data_type=TagDataType.BOOLEAN,
)

EMERGENCY_STOP = OPCUATag(
    name="emergency_stop",
    node_id="ns=2;s=Safety.EmergencyStop",
    category=TagCategory.SAFETY,
    data_type=TagDataType.BOOLEAN,
)

GUARD_OPEN = OPCUATag(
    name="guard_open",
    node_id="ns=2;s=Safety.GuardOpen",
    category=TagCategory.SAFETY,
    data_type=TagDataType.BOOLEAN,
)


# ---------------------------------------------------------
# Maintenance / health
# ---------------------------------------------------------

HEALTH_SCORE = OPCUATag(
    name="health_score",
    node_id="ns=2;s=Health.Score",
    category=TagCategory.HEALTH,
    data_type=TagDataType.FLOAT,
    unit="%",
    minimum=0.0,
    maximum=100.0,
    warning_low=60.0,
    critical_low=35.0,
)

FAILURE_PROBABILITY = OPCUATag(
    name="failure_probability",
    node_id="ns=2;s=Health.FailureProbability",
    category=TagCategory.HEALTH,
    data_type=TagDataType.FLOAT,
    minimum=0.0,
    maximum=1.0,
    warning_high=0.65,
    critical_high=0.85,
)

RUL_HOURS = OPCUATag(
    name="rul_hours",
    node_id="ns=2;s=Health.RULHours",
    category=TagCategory.HEALTH,
    data_type=TagDataType.FLOAT,
    unit="hours",
    warning_low=72.0,
    critical_low=24.0,
)

MAINTENANCE_REQUIRED = OPCUATag(
    name="maintenance_required",
    node_id="ns=2;s=Maintenance.Required",
    category=TagCategory.MAINTENANCE,
    data_type=TagDataType.BOOLEAN,
)


# ---------------------------------------------------------
# Authorized control tags
# ---------------------------------------------------------

START_COMMAND = OPCUATag(
    name="start_command",
    node_id="ns=2;s=Commands.Start",
    category=TagCategory.COMMAND,
    data_type=TagDataType.BOOLEAN,
    access=TagAccess.WRITE_ONLY,
)

STOP_COMMAND = OPCUATag(
    name="stop_command",
    node_id="ns=2;s=Commands.Stop",
    category=TagCategory.COMMAND,
    data_type=TagDataType.BOOLEAN,
    access=TagAccess.WRITE_ONLY,
)

RESET_ALARM = OPCUATag(
    name="reset_alarm",
    node_id="ns=2;s=Commands.ResetAlarm",
    category=TagCategory.COMMAND,
    data_type=TagDataType.BOOLEAN,
    access=TagAccess.WRITE_ONLY,
)


# ---------------------------------------------------------
# Registry
# ---------------------------------------------------------

ALL_TAGS: tuple[OPCUATag, ...] = (
    MACHINE_RUNNING,
    MACHINE_SPEED,
    MACHINE_LOAD,
    OPERATING_HOURS,

    BEARING_TEMPERATURE,
    MOTOR_TEMPERATURE,
    AMBIENT_TEMPERATURE,

    VIBRATION_RMS,
    VIBRATION_X,
    VIBRATION_Y,
    VIBRATION_Z,

    INLET_PRESSURE,
    OUTLET_PRESSURE,
    OIL_PRESSURE,

    CURRENT,
    VOLTAGE,
    POWER,
    POWER_FACTOR,

    ENERGY_CONSUMPTION,
    ENERGY_RATE,

    PRODUCTION_COUNT,
    PRODUCTION_RATE,
    SCRAP_COUNT,

    FIRE_DETECTED,
    SMOKE_DETECTED,
    OVERHEATING,
    EMERGENCY_STOP,
    GUARD_OPEN,

    HEALTH_SCORE,
    FAILURE_PROBABILITY,
    RUL_HOURS,
    MAINTENANCE_REQUIRED,

    START_COMMAND,
    STOP_COMMAND,
    RESET_ALARM,
)


TAG_REGISTRY: dict[str, OPCUATag] = {
    tag.name: tag
    for tag in ALL_TAGS
}


NODE_REGISTRY: dict[str, OPCUATag] = {
    tag.node_id: tag
    for tag in ALL_TAGS
}


# ---------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------

def get_tag(name: str) -> OPCUATag:
    try:
        return TAG_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown OPC-UA tag: {name}"
        ) from exc


def get_tag_by_node(
    node_id: str,
) -> OPCUATag:
    try:
        return NODE_REGISTRY[node_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown OPC-UA node: {node_id}"
        ) from exc


def get_tags_by_category(
    category: TagCategory,
) -> list[OPCUATag]:
    return [
        tag
        for tag in ALL_TAGS
        if tag.category == category
    ]


def readable_tags() -> list[OPCUATag]:
    return [
        tag
        for tag in ALL_TAGS
        if tag.readable
    ]


def writable_tags() -> list[OPCUATag]:
    return [
        tag
        for tag in ALL_TAGS
        if tag.writable
    ]


def node_ids(
    tags: Iterable[OPCUATag],
) -> list[str]:
    return [
        tag.node_id
        for tag in tags
    ]


# ---------------------------------------------------------
# Validation / alarm evaluation
# ---------------------------------------------------------

def validate_value(
    tag: OPCUATag,
    value: Any,
) -> bool:
    """Validate a value against basic tag constraints."""

    if tag.data_type == TagDataType.BOOLEAN:
        return isinstance(value, bool)

    if tag.data_type == TagDataType.INTEGER:
        if isinstance(value, bool):
            return False

        if not isinstance(value, int):
            return False

    elif tag.data_type == TagDataType.FLOAT:
        if isinstance(value, bool):
            return False

        if not isinstance(
            value,
            (int, float),
        ):
            return False

    elif tag.data_type == TagDataType.STRING:
        return isinstance(value, str)

    if isinstance(value, (int, float)):
        numeric = float(value)

        if (
            tag.minimum is not None
            and numeric < tag.minimum
        ):
            return False

        if (
            tag.maximum is not None
            and numeric > tag.maximum
        ):
            return False

    return True


def evaluate_threshold(
    tag: OPCUATag,
    value: float,
) -> str:
    """
    Evaluate telemetry against configured thresholds.

    Returns:
        normal
        warning_low
        warning_high
        critical_low
        critical_high
    """

    numeric = float(value)

    if (
        tag.critical_low is not None
        and numeric <= tag.critical_low
    ):
        return "critical_low"

    if (
        tag.critical_high is not None
        and numeric >= tag.critical_high
    ):
        return "critical_high"

    if (
        tag.warning_low is not None
        and numeric <= tag.warning_low
    ):
        return "warning_low"

    if (
        tag.warning_high is not None
        and numeric >= tag.warning_high
    ):
        return "warning_high"

    return "normal"


def telemetry_tags() -> list[OPCUATag]:
    """Return tags useful for the live telemetry pipeline."""

    categories = {
        TagCategory.MACHINE,
        TagCategory.TEMPERATURE,
        TagCategory.VIBRATION,
        TagCategory.PRESSURE,
        TagCategory.ELECTRICAL,
        TagCategory.ENERGY,
        TagCategory.PRODUCTION,
    }

    return [
        tag
        for tag in ALL_TAGS
        if tag.category in categories
        and tag.readable
    ]


def safety_tags() -> list[OPCUATag]:
    return get_tags_by_category(
        TagCategory.SAFETY
    )


def predictive_maintenance_tags() -> list[OPCUATag]:
    categories = {
        TagCategory.MACHINE,
        TagCategory.TEMPERATURE,
        TagCategory.VIBRATION,
        TagCategory.PRESSURE,
        TagCategory.ELECTRICAL,
        TagCategory.HEALTH,
        TagCategory.MAINTENANCE,
    }

    return [
        tag
        for tag in ALL_TAGS
        if tag.category in categories
        and tag.readable
    ]