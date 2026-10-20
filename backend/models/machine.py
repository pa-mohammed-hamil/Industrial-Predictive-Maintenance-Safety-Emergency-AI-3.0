"""
app/ai/ml/models/machine.py

Industrial Machine Intelligence Domain Models
==============================================

Python 3.11+
Pydantic >= 2.0

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant machine registration
- Machine and component relationships
- Manufacturer and technical specifications
- Machine operating state
- Sensor registration and telemetry snapshots
- Health score, anomaly and failure predictions
- Remaining useful life (RUL)
- Production performance and OEE
- Maintenance and inspection references
- Safety and emergency indicators
- Machine financial information
- Lifecycle management
- Create, update, response and summary schemas

This module defines domain models only.
It does not perform ML inference, database
operations or physical machine control.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
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


def require_aware(
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

class MachineType(str, Enum):
    CNC = "cnc"
    LATHE = "lathe"
    MILLING = "milling"
    DRILLING = "drilling"
    PRESS = "press"
    INJECTION_MOLDING = "injection_molding"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    MOTOR = "motor"
    GENERATOR = "generator"
    TURBINE = "turbine"
    CONVEYOR = "conveyor"
    ROBOT = "robot"
    BOILER = "boiler"
    CHILLER = "chiller"
    HVAC = "hvac"
    TRANSFORMER = "transformer"
    PACKAGING = "packaging"
    ASSEMBLY = "assembly"
    WELDING = "welding"
    FURNACE = "furnace"
    MIXER = "mixer"
    OTHER = "other"


class MachineStatus(str, Enum):
    REGISTERED = "registered"
    COMMISSIONING = "commissioning"
    ACTIVE = "active"
    IDLE = "idle"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    FAILED = "failed"
    OUT_OF_SERVICE = "out_of_service"
    DECOMMISSIONED = "decommissioned"


class MachineOperatingState(str, Enum):
    UNKNOWN = "unknown"
    RUNNING = "running"
    IDLE = "idle"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT = "fault"
    MAINTENANCE = "maintenance"


class MachineCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MachineHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MachineSensorType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    RPM = "rpm"
    FLOW = "flow"
    HUMIDITY = "humidity"
    ACOUSTIC = "acoustic"
    OIL_QUALITY = "oil_quality"
    POSITION = "position"
    TORQUE = "torque"
    OTHER = "other"


class MachineSafetyStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class MachineMaintenanceStrategy(str, Enum):
    REACTIVE = "reactive"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    RELIABILITY_CENTERED = "reliability_centered"


# ============================================================
# BASE MODEL
# ============================================================

class MachineBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# LOCATION
# ============================================================

class MachineLocation(MachineBaseModel):
    site_id: str = Field(min_length=1)
    site_name: str | None = None

    building: str | None = None
    floor: str | None = None

    production_line_id: str | None = None
    zone_id: str | None = None

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
    )
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
    )

    description: str | None = None


# ============================================================
# MANUFACTURER
# ============================================================

class MachineManufacturer(MachineBaseModel):
    manufacturer_name: str = Field(
        min_length=1,
        max_length=200,
    )
    model_number: str | None = None
    serial_number: str | None = None
    manufacturing_year: int | None = Field(
        default=None,
        ge=1900,
        le=2200,
    )
    country_of_origin: str | None = None


# ============================================================
# TECHNICAL SPECIFICATIONS
# ============================================================

class MachineSpecifications(MachineBaseModel):
    rated_power_kw: float | None = Field(
        default=None,
        ge=0,
    )
    rated_voltage_v: float | None = Field(
        default=None,
        ge=0,
    )
    rated_current_a: float | None = Field(
        default=None,
        ge=0,
    )
    rated_speed_rpm: float | None = Field(
        default=None,
        ge=0,
    )
    rated_pressure_bar: float | None = Field(
        default=None,
        ge=0,
    )
    maximum_temperature_c: float | None = None

    rated_capacity_units_per_hour: float | None = Field(
        default=None,
        ge=0,
    )

    expected_service_life_hours: float | None = Field(
        default=None,
        gt=0,
    )

    weight_kg: float | None = Field(
        default=None,
        ge=0,
    )

    additional_specifications: dict[
        str, Any
    ] = Field(default_factory=dict)

    @field_validator("maximum_temperature_c")
    @classmethod
    def validate_temperature(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError(
                "Temperature must be finite"
            )
        return value


# ============================================================
# SENSOR REGISTRATION
# ============================================================

class MachineSensor(MachineBaseModel):
    sensor_id: str = Field(min_length=1)
    sensor_type: MachineSensorType
    unit: str = Field(min_length=1)

    component_id: str | None = None

    minimum_operating_value: float | None = None
    maximum_operating_value: float | None = None

    installed_at: datetime | None = None
    last_calibrated_at: datetime | None = None

    active: bool = True

    @field_validator(
        "installed_at",
        "last_calibrated_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return require_aware(value)

    @model_validator(mode="after")
    def validate_limits(self) -> "MachineSensor":
        low = self.minimum_operating_value
        high = self.maximum_operating_value

        for value in (low, high):
            if value is not None and not isfinite(value):
                raise ValueError(
                    "Sensor limits must be finite"
                )

        if (
            low is not None
            and high is not None
            and low > high
        ):
            raise ValueError(
                "Minimum sensor value exceeds maximum"
            )

        return self


# ============================================================
# TELEMETRY SNAPSHOT
# ============================================================

class MachineTelemetrySnapshot(MachineBaseModel):
    sensor_id: str = Field(min_length=1)
    sensor_type: MachineSensorType

    value: float
    unit: str = Field(min_length=1)

    recorded_at: datetime = Field(
        default_factory=utc_now
    )

    quality: str = "unknown"
    source: str | None = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError(
                "Telemetry value must be finite"
            )
        return value

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(
        cls,
        value: datetime,
    ) -> datetime:
        return require_aware(value)


# ============================================================
# MACHINE HEALTH
# ============================================================

class MachineHealth(MachineBaseModel):
    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    health_status: MachineHealthStatus = (
        MachineHealthStatus.UNKNOWN
    )

    anomaly_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    failure_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    prediction_horizon_hours: float | None = Field(
        default=None,
        gt=0,
    )

    remaining_useful_life_hours: float | None = Field(
        default=None,
        ge=0,
    )

    visual_health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    latest_anomaly_prediction_id: str | None = None
    latest_failure_prediction_id: str | None = None
    latest_rul_prediction_id: str | None = None

    model_version: str | None = None

    evaluated_at: datetime | None = None

    @field_validator("evaluated_at")
    @classmethod
    def validate_evaluated_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return require_aware(value)

    @model_validator(mode="after")
    def validate_prediction_horizon(
        self,
    ) -> "MachineHealth":
        if (
            self.failure_probability is not None
            and self.prediction_horizon_hours is None
        ):
            raise ValueError(
                "Failure probability requires "
                "prediction_horizon_hours"
            )

        return self


# ============================================================
# MACHINE PRODUCTION METRICS
# ============================================================

class MachineProductionMetrics(MachineBaseModel):
    planned_production_minutes: float = Field(
        default=0,
        ge=0,
    )
    operating_minutes: float = Field(
        default=0,
        ge=0,
    )
    ideal_cycle_time_minutes: float | None = Field(
        default=None,
        gt=0,
    )
    total_units: int = Field(default=0, ge=0)
    good_units: int = Field(default=0, ge=0)
    rejected_units: int = Field(default=0, ge=0)

    measured_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_production(
        self,
    ) -> "MachineProductionMetrics":
        if (
            self.operating_minutes
            > self.planned_production_minutes
        ):
            raise ValueError(
                "Operating minutes exceed "
                "planned production minutes"
            )

        if (
            self.good_units + self.rejected_units
            > self.total_units
        ):
            raise ValueError(
                "Good and rejected units exceed total"
            )

        return self

    @property
    def availability(self) -> float | None:
        if self.planned_production_minutes == 0:
            return None

        return (
            self.operating_minutes
            / self.planned_production_minutes
        )

    @property
    def performance(self) -> float | None:
        if (
            self.operating_minutes == 0
            or self.ideal_cycle_time_minutes is None
        ):
            return None

        return (
            self.ideal_cycle_time_minutes
            * self.total_units
            / self.operating_minutes
        )

    @property
    def quality(self) -> float | None:
        if self.total_units == 0:
            return None

        return self.good_units / self.total_units

    @property
    def oee(self) -> float | None:
        if (
            self.availability is None
            or self.performance is None
            or self.quality is None
        ):
            return None

        return (
            self.availability
            * self.performance
            * self.quality
        )


# ============================================================
# MACHINE MAINTENANCE
# ============================================================

class MachineMaintenance(MachineBaseModel):
    strategy: MachineMaintenanceStrategy = (
        MachineMaintenanceStrategy.PREDICTIVE
    )

    last_maintenance_at: datetime | None = None
    next_maintenance_at: datetime | None = None

    open_work_order_count: int = Field(
        default=0,
        ge=0,
    )

    overdue_work_order_count: int = Field(
        default=0,
        ge=0,
    )

    maintenance_record_ids: list[str] = Field(
        default_factory=list
    )

    @field_validator(
        "last_maintenance_at",
        "next_maintenance_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return require_aware(value)


# ============================================================
# MACHINE SAFETY
# ============================================================

class MachineSafety(MachineBaseModel):
    safety_status: MachineSafetyStatus = (
        MachineSafetyStatus.UNKNOWN
    )

    open_incident_count: int = Field(
        default=0,
        ge=0,
    )

    active_emergency_count: int = Field(
        default=0,
        ge=0,
    )

    active_alert_count: int = Field(
        default=0,
        ge=0,
    )

    last_safety_inspection_at: datetime | None = None

    emergency_stop_reported: bool = False

    requires_engineering_review: bool = False

    @field_validator("last_safety_inspection_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return require_aware(value)


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

class MachineFinancials(MachineBaseModel):
    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    purchase_cost: Decimal | None = Field(
        default=None,
        ge=0,
    )

    replacement_cost: Decimal | None = Field(
        default=None,
        ge=0,
    )

    accumulated_maintenance_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    accumulated_downtime_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    accumulated_production_loss: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    estimated_future_cost: Decimal | None = Field(
        default=None,
        ge=0,
    )

    forecast_reference_id: str | None = None


# ============================================================
# MACHINE LIFECYCLE
# ============================================================

class MachineLifecycle(MachineBaseModel):
    purchased_at: datetime | None = None
    installed_at: datetime | None = None
    commissioned_at: datetime | None = None
    decommissioned_at: datetime | None = None

    operating_hours: float = Field(
        default=0,
        ge=0,
    )

    operating_cycles: int = Field(
        default=0,
        ge=0,
    )

    @field_validator(
        "purchased_at",
        "installed_at",
        "commissioned_at",
        "decommissioned_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return require_aware(value)


# ============================================================
# MAIN MACHINE MODEL
# ============================================================

class IndustrialMachine(MachineBaseModel):
    machine_id: UUID = Field(
        default_factory=uuid4
    )

    company_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)

    machine_code: str = Field(
        min_length=1,
        max_length=100,
    )

    machine_name: str = Field(
        min_length=1,
        max_length=250,
    )

    machine_type: MachineType

    description: str | None = None

    status: MachineStatus = (
        MachineStatus.REGISTERED
    )

    operating_state: MachineOperatingState = (
        MachineOperatingState.UNKNOWN
    )

    criticality: MachineCriticality = (
        MachineCriticality.MEDIUM
    )

    location: MachineLocation

    manufacturer: MachineManufacturer | None = None

    specifications: MachineSpecifications = Field(
        default_factory=MachineSpecifications
    )

    sensors: list[MachineSensor] = Field(
        default_factory=list
    )

    latest_telemetry: list[
        MachineTelemetrySnapshot
    ] = Field(default_factory=list)

    component_ids: list[str] = Field(
        default_factory=list
    )

    health: MachineHealth = Field(
        default_factory=MachineHealth
    )

    production: MachineProductionMetrics = Field(
        default_factory=MachineProductionMetrics
    )

    maintenance: MachineMaintenance = Field(
        default_factory=MachineMaintenance
    )

    safety: MachineSafety = Field(
        default_factory=MachineSafety
    )

    financials: MachineFinancials = Field(
        default_factory=MachineFinancials
    )

    lifecycle: MachineLifecycle = Field(
        default_factory=MachineLifecycle
    )

    latest_inspection_id: str | None = None
    latest_failure_id: str | None = None
    latest_incident_id: str | None = None

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
        "created_at",
        "updated_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime,
    ) -> datetime:
        return require_aware(value)

    @model_validator(mode="after")
    def validate_unique_sensors(
        self,
    ) -> "IndustrialMachine":
        sensor_ids = [
            sensor.sensor_id
            for sensor in self.sensors
        ]

        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError(
                "Duplicate sensor IDs"
            )

        if len(self.component_ids) != len(
            set(self.component_ids)
        ):
            raise ValueError(
                "Duplicate component IDs"
            )

        return self

    def _touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["requires_engineering_review"] = (
            self.needs_engineering_review()
        )
        return data

    # --------------------------------------------------------
    # SENSOR MANAGEMENT
    # --------------------------------------------------------

    def add_sensor(
        self,
        sensor: MachineSensor,
    ) -> None:
        if any(
            existing.sensor_id == sensor.sensor_id
            for existing in self.sensors
        ):
            raise ValueError(
                "Sensor already registered"
            )

        self.sensors.append(sensor)
        self._touch()

    def add_component(
        self,
        component_id: str,
    ) -> None:
        if not component_id.strip():
            raise ValueError(
                "component_id is required"
            )

        if component_id in self.component_ids:
            raise ValueError(
                "Component already registered"
            )

        self.component_ids.append(component_id)
        self._touch()

    # --------------------------------------------------------
    # TELEMETRY
    # --------------------------------------------------------

    def update_telemetry(
        self,
        reading: MachineTelemetrySnapshot,
    ) -> None:
        sensor = next(
            (
                item
                for item in self.sensors
                if item.sensor_id == reading.sensor_id
            ),
            None,
        )

        if sensor is None:
            raise ValueError(
                "Telemetry references unknown sensor"
            )

        if not sensor.active:
            raise ValueError(
                "Sensor is inactive"
            )

        if (
            sensor.sensor_type != reading.sensor_type
            or sensor.unit != reading.unit
        ):
            raise ValueError(
                "Telemetry sensor type or unit mismatch"
            )

        existing = next(
            (
                item
                for item in self.latest_telemetry
                if item.sensor_id == reading.sensor_id
            ),
            None,
        )

        if existing is not None:
            if reading.recorded_at <= existing.recorded_at:
                raise ValueError(
                    "Telemetry must be newer "
                    "than the existing reading"
                )

            self.latest_telemetry.remove(existing)

        self.latest_telemetry.append(reading)
        self._touch()

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    def update_health(
        self,
        health: MachineHealth,
    ) -> None:
        if (
            self.health.evaluated_at is not None
            and health.evaluated_at is not None
            and health.evaluated_at
            < self.health.evaluated_at
        ):
            raise ValueError(
                "Cannot overwrite health "
                "with an older assessment"
            )

        self.health = health
        self._touch()

    # --------------------------------------------------------
    # MACHINE STATUS
    # --------------------------------------------------------

    def set_status(
        self,
        status: MachineStatus,
    ) -> None:
        if (
            self.status
            == MachineStatus.DECOMMISSIONED.value
        ):
            raise ValueError(
                "Decommissioned machine cannot "
                "change status"
            )

        if status == MachineStatus.DECOMMISSIONED:
            raise ValueError(
                "Use decommission() for this transition"
            )

        self.status = status
        self._touch()

    def decommission(self) -> None:
        if (
            self.operating_state
            == MachineOperatingState.RUNNING.value
        ):
            raise ValueError(
                "Cannot decommission a machine "
                "reported as running"
            )

        self.status = MachineStatus.DECOMMISSIONED
        self.operating_state = (
            MachineOperatingState.STOPPED
        )
        self.lifecycle.decommissioned_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # ENGINEERING REVIEW
    # --------------------------------------------------------

    def needs_engineering_review(self) -> bool:
        return any([
            self.health.health_status
            == MachineHealthStatus.CRITICAL.value,

            self.safety.safety_status in {
                MachineSafetyStatus.CRITICAL.value,
                MachineSafetyStatus.EMERGENCY.value,
            },

            self.safety.active_emergency_count > 0,

            self.safety.requires_engineering_review,

            self.status
            == MachineStatus.FAILED.value,
        ])


# ============================================================
# CREATE SCHEMA
# ============================================================

class MachineCreate(MachineBaseModel):
    company_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)

    machine_code: str = Field(min_length=1)
    machine_name: str = Field(min_length=1)

    machine_type: MachineType

    description: str | None = None

    criticality: MachineCriticality = (
        MachineCriticality.MEDIUM
    )

    location: MachineLocation

    manufacturer: MachineManufacturer | None = None

    specifications: MachineSpecifications = Field(
        default_factory=MachineSpecifications
    )


# ============================================================
# UPDATE SCHEMA
# ============================================================

class MachineUpdate(MachineBaseModel):
    machine_name: str | None = None
    description: str | None = None

    criticality: MachineCriticality | None = None

    location: MachineLocation | None = None

    specifications: MachineSpecifications | None = None

    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class MachineResponse(IndustrialMachine):
    pass


# ============================================================
# SUMMARY
# ============================================================

class MachineSummary(MachineBaseModel):
    machine_id: UUID

    company_id: str
    asset_id: str

    machine_code: str
    machine_name: str

    machine_type: MachineType
    status: MachineStatus
    operating_state: MachineOperatingState

    criticality: MachineCriticality

    health_score: float | None
    health_status: MachineHealthStatus

    failure_probability: float | None
    remaining_useful_life_hours: float | None

    sensor_count: int
    component_count: int

    requires_engineering_review: bool

    updated_at: datetime

    @classmethod
    def from_machine(
        cls,
        machine: IndustrialMachine,
    ) -> "MachineSummary":
        return cls(
            machine_id=machine.machine_id,
            company_id=machine.company_id,
            asset_id=machine.asset_id,
            machine_code=machine.machine_code,
            machine_name=machine.machine_name,
            machine_type=machine.machine_type,
            status=machine.status,
            operating_state=machine.operating_state,
            criticality=machine.criticality,
            health_score=machine.health.health_score,
            health_status=machine.health.health_status,
            failure_probability=(
                machine.health.failure_probability
            ),
            remaining_useful_life_hours=(
                machine.health.remaining_useful_life_hours
            ),
            sensor_count=len(machine.sensors),
            component_count=len(machine.component_ids),
            requires_engineering_review=(
                machine.needs_engineering_review()
            ),
            updated_at=machine.updated_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    machine = IndustrialMachine(
        company_id="company-001",
        asset_id="ASSET-101",
        machine_code="PUMP-101",
        machine_name="Industrial Cooling Pump",
        machine_type=MachineType.PUMP,
        status=MachineStatus.ACTIVE,
        operating_state=MachineOperatingState.RUNNING,
        criticality=MachineCriticality.HIGH,
        location=MachineLocation(
            site_id="SITE-01",
            production_line_id="LINE-01",
            zone_id="PUMP-ROOM-A",
        ),
        manufacturer=MachineManufacturer(
            manufacturer_name="Example Industrial",
            model_number="P-500",
        ),
        specifications=MachineSpecifications(
            rated_power_kw=75,
            rated_speed_rpm=1500,
            expected_service_life_hours=30000,
        ),
        sensors=[
            MachineSensor(
                sensor_id="TEMP-101",
                sensor_type=MachineSensorType.TEMPERATURE,
                unit="C",
            ),
            MachineSensor(
                sensor_id="VIB-101",
                sensor_type=MachineSensorType.VIBRATION,
                unit="mm/s",
            ),
        ],
    )

    machine.update_telemetry(
        MachineTelemetrySnapshot(
            sensor_id="TEMP-101",
            sensor_type=MachineSensorType.TEMPERATURE,
            value=82.5,
            unit="C",
        )
    )

    # Illustrative prediction values only.
    machine.update_health(
        MachineHealth(
            health_score=64,
            health_status=MachineHealthStatus.DEGRADED,
            anomaly_score=0.72,
            failure_probability=0.35,
            prediction_horizon_hours=168,
            remaining_useful_life_hours=1200,
            model_version="1.0.0",
            evaluated_at=utc_now(),
        )
    )

    print(
        json.dumps(
            MachineSummary.from_machine(
                machine
            ).model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()