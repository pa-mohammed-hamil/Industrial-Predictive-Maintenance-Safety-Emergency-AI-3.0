"""
app/ai/ml/models/sensor.py

Industrial IoT Sensor Intelligence Models
=========================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant sensor registry
- Industrial sensor types
- MQTT, OPC-UA, Modbus, SCADA and HTTP sources
- Sensor installation and configuration
- Sensor calibration
- Sensor health monitoring
- Sensor telemetry readings
- Warning and critical thresholds
- Sensor anomaly records
- Sensor alerts
- Sensor lifecycle management
- Create, update and response schemas
- Sensor summaries

This module provides validated domain models.

It does not connect to physical sensors, execute ML
models, send alerts or control industrial equipment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_aware(
    value: datetime | None,
) -> datetime | None:
    if value is not None and (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "Datetime must be timezone-aware"
        )

    return value


# ============================================================
# ENUMS
# ============================================================

class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    VOLTAGE = "voltage"
    HUMIDITY = "humidity"

    GAS = "gas"
    SMOKE = "smoke"
    FIRE = "fire"

    FLOW = "flow"
    LEVEL = "level"
    RPM = "rpm"
    SPEED = "speed"

    ACCELERATION = "acceleration"
    ACOUSTIC = "acoustic"
    ULTRASONIC = "ultrasonic"

    TORQUE = "torque"
    FORCE = "force"
    LOAD = "load"

    POWER = "power"
    ENERGY = "energy"

    PROXIMITY = "proximity"
    MOTION = "motion"

    AIR_QUALITY = "air_quality"
    PARTICULATE = "particulate"

    OIL_QUALITY = "oil_quality"
    PH = "ph"

    OTHER = "other"


class SensorStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    FAULTY = "faulty"
    OFFLINE = "offline"
    DECOMMISSIONED = "decommissioned"


class SensorHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class SensorProtocol(str, Enum):
    MQTT = "mqtt"
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    KAFKA = "kafka"
    SCADA = "scada"
    PLC = "plc"
    SERIAL = "serial"
    BLE = "ble"
    LORAWAN = "lorawan"
    OTHER = "other"


class SensorReadingQuality(str, Enum):
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    MISSING = "missing"
    OUT_OF_RANGE = "out_of_range"
    STALE = "stale"


class SensorAlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class SensorAlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SensorThresholdOperator(str, Enum):
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class SensorCalibrationStatus(str, Enum):
    NOT_CALIBRATED = "not_calibrated"
    VALID = "valid"
    DUE = "due"
    EXPIRED = "expired"
    FAILED = "failed"


class SensorAnomalyType(str, Enum):
    SPIKE = "spike"
    DROP = "drop"
    DRIFT = "drift"
    OUTLIER = "outlier"
    STUCK_VALUE = "stuck_value"
    NOISE = "noise"
    MISSING_DATA = "missing_data"
    SENSOR_FAILURE = "sensor_failure"
    UNKNOWN = "unknown"


# ============================================================
# BASE MODEL
# ============================================================

class SensorBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ============================================================
# SENSOR CONTEXT
# ============================================================

class SensorContext(SensorBaseModel):
    company_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)

    asset_id: str | None = None
    machine_id: str | None = None
    equipment_id: str | None = None
    component_id: str | None = None

    production_line_id: str | None = None
    zone_id: str | None = None

    building: str | None = None
    floor: str | None = None

    installation_location: str | None = None


# ============================================================
# SENSOR MANUFACTURER
# ============================================================

class SensorManufacturer(SensorBaseModel):
    manufacturer_name: str = Field(min_length=1)

    model_number: str | None = None
    serial_number: str | None = None

    firmware_version: str | None = None
    hardware_version: str | None = None

    country_of_origin: str | None = None


# ============================================================
# SENSOR CONNECTION
# ============================================================

class SensorConnection(SensorBaseModel):
    protocol: SensorProtocol

    gateway_id: str | None = None
    endpoint_reference: str | None = None

    mqtt_topic: str | None = None
    kafka_topic: str | None = None

    device_address: str | None = None
    register_address: str | None = None

    polling_interval_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    sampling_frequency_hz: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    secure_connection_required: bool = True

    credential_reference: str | None = None

    @model_validator(mode="after")
    def validate_connection(
        self,
    ) -> "SensorConnection":
        if (
            self.protocol == SensorProtocol.MQTT.value
            and not self.mqtt_topic
        ):
            raise ValueError(
                "MQTT sensors require mqtt_topic"
            )

        if (
            self.protocol == SensorProtocol.KAFKA.value
            and not self.kafka_topic
        ):
            raise ValueError(
                "Kafka sensors require kafka_topic"
            )

        return self


# ============================================================
# SENSOR SPECIFICATIONS
# ============================================================

class SensorSpecifications(SensorBaseModel):
    measurement_unit: str = Field(min_length=1)

    minimum_range: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    maximum_range: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    accuracy_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    resolution: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    response_time_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    operating_temperature_min_c: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    operating_temperature_max_c: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_ranges(
        self,
    ) -> "SensorSpecifications":
        if (
            self.minimum_range is not None
            and self.maximum_range is not None
            and self.minimum_range >= self.maximum_range
        ):
            raise ValueError(
                "minimum_range must be below maximum_range"
            )

        if (
            self.operating_temperature_min_c is not None
            and self.operating_temperature_max_c is not None
            and self.operating_temperature_min_c
            >= self.operating_temperature_max_c
        ):
            raise ValueError(
                "Invalid operating temperature range"
            )

        return self


# ============================================================
# SENSOR CALIBRATION
# ============================================================

class SensorCalibration(SensorBaseModel):
    calibration_id: UUID = Field(
        default_factory=uuid4
    )

    status: SensorCalibrationStatus = (
        SensorCalibrationStatus.NOT_CALIBRATED
    )

    calibrated_at: datetime | None = None
    next_calibration_at: datetime | None = None

    calibrated_by: str | None = None

    calibration_certificate_reference: str | None = None

    offset: float = Field(
        default=0,
        allow_inf_nan=False,
    )

    scale_factor: float = Field(
        default=1,
        gt=0,
        allow_inf_nan=False,
    )

    notes: str | None = None

    @field_validator(
        "calibrated_at",
        "next_calibration_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_calibration(
        self,
    ) -> "SensorCalibration":
        if (
            self.calibrated_at is not None
            and self.next_calibration_at is not None
            and self.next_calibration_at
            <= self.calibrated_at
        ):
            raise ValueError(
                "Next calibration must follow calibration"
            )

        if (
            self.status == SensorCalibrationStatus.VALID.value
            and not (
                self.calibrated_at
                and self.calibrated_by
            )
        ):
            raise ValueError(
                "Valid calibration requires "
                "date and calibrator"
            )

        return self

    def is_due(
        self,
        reference_time: datetime | None = None,
    ) -> bool:
        if self.next_calibration_at is None:
            return False

        now = reference_time or utc_now()
        validate_aware(now)

        return now >= self.next_calibration_at


# ============================================================
# SENSOR THRESHOLD
# ============================================================

class SensorThreshold(SensorBaseModel):
    threshold_id: UUID = Field(
        default_factory=uuid4
    )

    name: str = Field(min_length=1)

    operator: SensorThresholdOperator

    threshold_value: float = Field(
        allow_inf_nan=False,
    )

    severity: SensorAlertSeverity

    enabled: bool = True

    description: str | None = None

    def is_triggered(self, value: float) -> bool:
        if not self.enabled:
            return False

        if not isfinite(value):
            raise ValueError(
                "Sensor value must be finite"
            )

        threshold = self.threshold_value

        operations = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }

        return operations[self.operator](
            value,
            threshold,
        )


# ============================================================
# SENSOR READING
# ============================================================

class SensorReading(SensorBaseModel):
    reading_id: UUID = Field(
        default_factory=uuid4
    )

    sensor_id: str = Field(min_length=1)

    value: float = Field(
        allow_inf_nan=False,
    )

    unit: str = Field(min_length=1)

    quality: SensorReadingQuality = (
        SensorReadingQuality.GOOD
    )

    recorded_at: datetime = Field(
        default_factory=utc_now
    )

    received_at: datetime = Field(
        default_factory=utc_now
    )

    source_message_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator(
        "recorded_at",
        "received_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_reading(
        self,
    ) -> "SensorReading":
        if self.received_at < self.recorded_at:
            raise ValueError(
                "received_at cannot precede recorded_at"
            )

        return self


# ============================================================
# SENSOR HEALTH
# ============================================================

class SensorHealth(SensorBaseModel):
    status: SensorHealthStatus = (
        SensorHealthStatus.UNKNOWN
    )

    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    signal_strength_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    battery_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    packet_loss_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    data_quality_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    last_seen_at: datetime | None = None

    last_diagnostic_at: datetime | None = None

    error_codes: list[str] = Field(
        default_factory=list
    )

    notes: str | None = None

    @field_validator(
        "last_seen_at",
        "last_diagnostic_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)


# ============================================================
# SENSOR ANOMALY
# ============================================================

class SensorAnomaly(SensorBaseModel):
    anomaly_id: UUID = Field(
        default_factory=uuid4
    )

    anomaly_type: SensorAnomalyType

    anomaly_score: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    description: str | None = None

    model_id: str | None = None
    model_version: str | None = None

    detected_at: datetime = Field(
        default_factory=utc_now
    )

    requires_review: bool = True

    @field_validator("detected_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_aware(value)


# ============================================================
# SENSOR ALERT
# ============================================================

class SensorAlert(SensorBaseModel):
    alert_id: UUID = Field(
        default_factory=uuid4
    )

    threshold_id: UUID | None = None

    severity: SensorAlertSeverity

    title: str = Field(min_length=1)

    message: str = Field(min_length=1)

    status: SensorAlertStatus = (
        SensorAlertStatus.OPEN
    )

    reading_id: UUID | None = None

    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    resolved_by: str | None = None
    resolved_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator(
        "acknowledged_at",
        "resolved_at",
        "created_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_alert(
        self,
    ) -> "SensorAlert":
        if (
            self.status
            == SensorAlertStatus.ACKNOWLEDGED.value
            and not (
                self.acknowledged_by
                and self.acknowledged_at
            )
        ):
            raise ValueError(
                "Acknowledged alert requires actor and time"
            )

        if self.status in {
            SensorAlertStatus.RESOLVED.value,
            SensorAlertStatus.CLOSED.value,
        }:
            if not (
                self.resolved_by
                and self.resolved_at
            ):
                raise ValueError(
                    "Resolved alert requires actor and time"
                )

        return self


# ============================================================
# MAIN INDUSTRIAL SENSOR
# ============================================================

class IndustrialSensor(SensorBaseModel):
    sensor_id: str = Field(min_length=1)

    sensor_uuid: UUID = Field(
        default_factory=uuid4
    )

    sensor_name: str = Field(
        min_length=1,
        max_length=250,
    )

    sensor_type: SensorType

    context: SensorContext

    manufacturer: SensorManufacturer | None = None

    connection: SensorConnection

    specifications: SensorSpecifications

    status: SensorStatus = SensorStatus.REGISTERED

    health: SensorHealth = Field(
        default_factory=SensorHealth
    )

    calibration: SensorCalibration = Field(
        default_factory=SensorCalibration
    )

    thresholds: list[SensorThreshold] = Field(
        default_factory=list
    )

    latest_reading: SensorReading | None = None

    anomalies: list[SensorAnomaly] = Field(
        default_factory=list
    )

    alerts: list[SensorAlert] = Field(
        default_factory=list
    )

    installed_at: datetime | None = None

    activated_at: datetime | None = None

    decommissioned_at: datetime | None = None

    tags: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator(
        "installed_at",
        "activated_at",
        "decommissioned_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_sensor(
        self,
    ) -> "IndustrialSensor":
        threshold_ids = [
            item.threshold_id
            for item in self.thresholds
        ]

        if len(threshold_ids) != len(
            set(threshold_ids)
        ):
            raise ValueError(
                "Duplicate sensor threshold IDs"
            )

        if (
            self.latest_reading is not None
            and self.latest_reading.sensor_id
            != self.sensor_id
        ):
            raise ValueError(
                "Latest reading belongs to another sensor"
            )

        return self

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = utc_now()

    # --------------------------------------------------------
    # ACTIVATE
    # --------------------------------------------------------

    def activate(self) -> None:
        if self.status == SensorStatus.DECOMMISSIONED.value:
            raise ValueError(
                "Decommissioned sensor cannot be activated"
            )

        self.status = SensorStatus.ACTIVE
        self.activated_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # DEACTIVATE
    # --------------------------------------------------------

    def deactivate(self) -> None:
        if self.status == SensorStatus.DECOMMISSIONED.value:
            raise ValueError(
                "Sensor is already decommissioned"
            )

        self.status = SensorStatus.INACTIVE
        self._touch()

    # --------------------------------------------------------
    # DECOMMISSION
    # --------------------------------------------------------

    def decommission(self) -> None:
        self.status = SensorStatus.DECOMMISSIONED
        self.decommissioned_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # ADD THRESHOLD
    # --------------------------------------------------------

    def add_threshold(
        self,
        threshold: SensorThreshold,
    ) -> None:
        if any(
            item.threshold_id == threshold.threshold_id
            for item in self.thresholds
        ):
            raise ValueError(
                "Threshold already exists"
            )

        self.thresholds.append(threshold)
        self._touch()

    # --------------------------------------------------------
    # RECORD READING
    # --------------------------------------------------------

    def record_reading(
        self,
        reading: SensorReading,
    ) -> list[SensorAlert]:
        if self.status != SensorStatus.ACTIVE.value:
            raise ValueError(
                "Sensor must be active"
            )

        if reading.sensor_id != self.sensor_id:
            raise ValueError(
                "Reading sensor ID mismatch"
            )

        if (
            reading.unit
            != self.specifications.measurement_unit
        ):
            raise ValueError(
                "Reading unit does not match sensor unit"
            )

        if (
            self.latest_reading is not None
            and reading.recorded_at
            < self.latest_reading.recorded_at
        ):
            raise ValueError(
                "Out-of-order reading; ingest historical "
                "telemetry through the telemetry service"
            )

        if reading.quality != SensorReadingQuality.GOOD.value:
            self.latest_reading = reading
            self.health.last_seen_at = reading.received_at
            self._touch()
            return []

        self.latest_reading = reading
        self.health.last_seen_at = reading.received_at

        generated_alerts: list[SensorAlert] = []

        for threshold in self.thresholds:
            if threshold.is_triggered(reading.value):
                alert = SensorAlert(
                    threshold_id=threshold.threshold_id,
                    severity=threshold.severity,
                    title=(
                        f"{self.sensor_name}: "
                        f"{threshold.name}"
                    ),
                    message=(
                        f"Sensor value {reading.value} "
                        f"{reading.unit} triggered "
                        f"{threshold.operator} "
                        f"{threshold.threshold_value}"
                    ),
                    reading_id=reading.reading_id,
                )

                self.alerts.append(alert)
                generated_alerts.append(alert)

        self._touch()
        return generated_alerts

    # --------------------------------------------------------
    # ADD ANOMALY
    # --------------------------------------------------------

    def add_anomaly(
        self,
        anomaly: SensorAnomaly,
    ) -> None:
        if any(
            item.anomaly_id == anomaly.anomaly_id
            for item in self.anomalies
        ):
            raise ValueError(
                "Duplicate anomaly ID"
            )

        self.anomalies.append(anomaly)
        self._touch()

    # --------------------------------------------------------
    # UPDATE HEALTH
    # --------------------------------------------------------

    def update_health(
        self,
        health: SensorHealth,
    ) -> None:
        self.health = health
        self._touch()

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    def update_calibration(
        self,
        calibration: SensorCalibration,
    ) -> None:
        self.calibration = calibration
        self._touch()

    # --------------------------------------------------------
    # ALERT HELPERS
    # --------------------------------------------------------

    def open_alert_count(self) -> int:
        return sum(
            alert.status in {
                SensorAlertStatus.OPEN.value,
                SensorAlertStatus.ACKNOWLEDGED.value,
            }
            for alert in self.alerts
        )

    def critical_alert_count(self) -> int:
        return sum(
            alert.severity
            == SensorAlertSeverity.CRITICAL.value
            and alert.status
            not in {
                SensorAlertStatus.RESOLVED.value,
                SensorAlertStatus.CLOSED.value,
            }
            for alert in self.alerts
        )

    def requires_maintenance(self) -> bool:
        return (
            self.health.status in {
                SensorHealthStatus.DEGRADED.value,
                SensorHealthStatus.WARNING.value,
                SensorHealthStatus.CRITICAL.value,
                SensorHealthStatus.FAILED.value,
            }
            or self.calibration.is_due()
            or self.critical_alert_count() > 0
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class SensorCreate(SensorBaseModel):
    sensor_id: str = Field(min_length=1)
    sensor_name: str = Field(min_length=1)

    sensor_type: SensorType

    context: SensorContext

    manufacturer: SensorManufacturer | None = None

    connection: SensorConnection

    specifications: SensorSpecifications

    installed_at: datetime | None = None


# ============================================================
# UPDATE SCHEMA
# ============================================================

class SensorUpdate(SensorBaseModel):
    sensor_name: str | None = None

    connection: SensorConnection | None = None

    specifications: SensorSpecifications | None = None

    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class SensorResponse(IndustrialSensor):
    pass


# ============================================================
# SUMMARY
# ============================================================

class SensorSummary(SensorBaseModel):
    sensor_id: str

    sensor_name: str

    sensor_type: SensorType

    company_id: str
    site_id: str

    asset_id: str | None
    machine_id: str | None

    status: SensorStatus

    health_status: SensorHealthStatus

    latest_value: float | None
    measurement_unit: str

    open_alert_count: int
    critical_alert_count: int

    calibration_due: bool
    requires_maintenance: bool

    updated_at: datetime

    @classmethod
    def from_sensor(
        cls,
        sensor: IndustrialSensor,
    ) -> "SensorSummary":
        return cls(
            sensor_id=sensor.sensor_id,
            sensor_name=sensor.sensor_name,
            sensor_type=sensor.sensor_type,
            company_id=sensor.context.company_id,
            site_id=sensor.context.site_id,
            asset_id=sensor.context.asset_id,
            machine_id=sensor.context.machine_id,
            status=sensor.status,
            health_status=sensor.health.status,
            latest_value=(
                sensor.latest_reading.value
                if sensor.latest_reading is not None
                else None
            ),
            measurement_unit=(
                sensor.specifications.measurement_unit
            ),
            open_alert_count=sensor.open_alert_count(),
            critical_alert_count=(
                sensor.critical_alert_count()
            ),
            calibration_due=(
                sensor.calibration.is_due()
            ),
            requires_maintenance=(
                sensor.requires_maintenance()
            ),
            updated_at=sensor.updated_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    sensor = IndustrialSensor(
        sensor_id="TEMP-101",
        sensor_name="Pump Motor Temperature Sensor",
        sensor_type=SensorType.TEMPERATURE,
        context=SensorContext(
            company_id="company-001",
            site_id="SITE-01",
            asset_id="ASSET-101",
            machine_id="PUMP-101",
            component_id="MOTOR-101",
        ),
        manufacturer=SensorManufacturer(
            manufacturer_name="Industrial Sensor Co.",
            model_number="TMP-100",
        ),
        connection=SensorConnection(
            protocol=SensorProtocol.MQTT,
            gateway_id="GATEWAY-001",
            mqtt_topic=(
                "factory/site-01/pump-101/temperature"
            ),
            sampling_frequency_hz=1,
        ),
        specifications=SensorSpecifications(
            measurement_unit="C",
            minimum_range=-20,
            maximum_range=200,
            accuracy_percent=1,
        ),
    )

    sensor.add_threshold(
        SensorThreshold(
            name="High Temperature",
            operator=SensorThresholdOperator.GREATER_THAN,
            threshold_value=85,
            severity=SensorAlertSeverity.WARNING,
        )
    )

    sensor.add_threshold(
        SensorThreshold(
            name="Critical Temperature",
            operator=SensorThresholdOperator.GREATER_THAN,
            threshold_value=100,
            severity=SensorAlertSeverity.CRITICAL,
        )
    )

    sensor.activate()

    alerts = sensor.record_reading(
        SensorReading(
            sensor_id="TEMP-101",
            value=105.5,
            unit="C",
        )
    )

    print("Generated alerts:", len(alerts))

    print(
        json.dumps(
            SensorSummary.from_sensor(
                sensor
            ).to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()