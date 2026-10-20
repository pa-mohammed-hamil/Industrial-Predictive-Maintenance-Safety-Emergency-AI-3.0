
"""
app/ai/ml/models/asset.py

Industrial Asset Data Models
============================

Python 3.11+
Pydantic v2

Dependencies:
    pip install "pydantic>=2,<3"

Features:
- Industrial asset identification
- Company and site ownership
- Equipment classifications
- Asset lifecycle tracking
- Operational status
- Asset health indicators
- Sensor configuration
- Maintenance metadata
- Warranty information
- Financial information
- ML prediction references
- Inspection information
- JSON-compatible schemas

This module defines data contracts.
It does not train or execute ML models.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
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
# ENUMS
# ============================================================

class AssetType(str, Enum):
    MOTOR = "motor"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    TURBINE = "turbine"
    GENERATOR = "generator"
    TRANSFORMER = "transformer"
    CONVEYOR = "conveyor"
    BOILER = "boiler"
    HVAC = "hvac"
    ROBOT = "robot"
    CNC_MACHINE = "cnc_machine"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    VALVE = "valve"
    FAN = "fan"
    CHILLER = "chiller"
    HYDRAULIC_SYSTEM = "hydraulic_system"
    ELECTRICAL_PANEL = "electrical_panel"
    OTHER = "other"


class AssetStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    UNDER_MAINTENANCE = "under_maintenance"
    DEGRADED = "degraded"
    FAULT = "fault"
    OUT_OF_SERVICE = "out_of_service"
    DECOMMISSIONED = "decommissioned"


class AssetCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssetHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    VOLTAGE = "voltage"
    RPM = "rpm"
    FLOW = "flow"
    HUMIDITY = "humidity"
    ACOUSTIC = "acoustic"
    THERMAL = "thermal"
    OTHER = "other"


class MaintenanceStrategy(str, Enum):
    REACTIVE = "reactive"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"


# ============================================================
# BASE MODEL
# ============================================================

class AssetBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# LOCATION
# ============================================================

class AssetLocation(AssetBaseModel):
    site_id: str = Field(
        min_length=1,
        max_length=100,
    )

    site_name: str | None = None

    building: str | None = None
    floor: str | None = None
    production_line: str | None = None
    zone: str | None = None

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


# ============================================================
# MANUFACTURER
# ============================================================

class AssetManufacturer(AssetBaseModel):
    manufacturer_name: str = Field(
        min_length=1,
        max_length=200,
    )

    model_number: str | None = None
    serial_number: str | None = None
    country_of_origin: str | None = None

    support_contact: str | None = None


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

class AssetSensor(AssetBaseModel):
    sensor_id: str = Field(
        min_length=1,
        max_length=100,
    )

    sensor_type: SensorType

    unit: str = Field(
        min_length=1,
        max_length=30,
    )

    enabled: bool = True

    minimum_expected_value: float | None = None
    maximum_expected_value: float | None = None

    warning_threshold: float | None = None
    critical_threshold: float | None = None

    sampling_interval_seconds: float = Field(
        default=1.0,
        gt=0,
    )

    last_calibrated_at: datetime | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "AssetSensor":
        if (
            self.minimum_expected_value is not None
            and self.maximum_expected_value is not None
            and self.minimum_expected_value
            >= self.maximum_expected_value
        ):
            raise ValueError(
                "Sensor minimum must be below maximum"
            )

        return self


# ============================================================
# ASSET HEALTH
# ============================================================

class AssetHealth(AssetBaseModel):
    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    health_status: AssetHealthStatus = (
        AssetHealthStatus.UNKNOWN
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

    remaining_useful_life_hours: float | None = (
        Field(
            default=None,
            ge=0,
        )
    )

    visual_health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    latest_prediction_id: str | None = None

    last_evaluated_at: datetime | None = None

    model_version: str | None = None


# ============================================================
# MAINTENANCE
# ============================================================

class AssetMaintenance(AssetBaseModel):
    strategy: MaintenanceStrategy = (
        MaintenanceStrategy.PREVENTIVE
    )

    last_maintenance_date: date | None = None
    next_maintenance_date: date | None = None

    maintenance_interval_days: int | None = Field(
        default=None,
        gt=0,
    )

    total_maintenance_events: int = Field(
        default=0,
        ge=0,
    )

    open_work_orders: int = Field(
        default=0,
        ge=0,
    )

    maintenance_notes: str | None = None


# ============================================================
# WARRANTY
# ============================================================

class AssetWarranty(AssetBaseModel):
    warranty_provider: str | None = None

    warranty_start_date: date | None = None
    warranty_end_date: date | None = None

    warranty_reference: str | None = None

    coverage_description: str | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "AssetWarranty":
        if (
            self.warranty_start_date
            and self.warranty_end_date
            and self.warranty_end_date
            < self.warranty_start_date
        ):
            raise ValueError(
                "Warranty end date cannot precede start"
            )

        return self


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

class AssetFinancials(AssetBaseModel):
    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    purchase_cost: float | None = Field(
        default=None,
        ge=0,
    )

    replacement_cost: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_current_value: float | None = (
        Field(
            default=None,
            ge=0,
        )
    )

    annual_maintenance_cost: float | None = (
        Field(
            default=None,
            ge=0,
        )
    )

    estimated_downtime_cost_per_hour: (
        float | None
    ) = Field(
        default=None,
        ge=0,
    )


# ============================================================
# OPERATIONAL METRICS
# ============================================================

class AssetOperationalMetrics(AssetBaseModel):
    operating_hours: float = Field(
        default=0,
        ge=0,
    )

    operating_cycles: int = Field(
        default=0,
        ge=0,
    )

    utilization_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    availability_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    performance_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    quality_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    oee_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    last_started_at: datetime | None = None
    last_stopped_at: datetime | None = None


# ============================================================
# MAIN ASSET MODEL
# ============================================================

class IndustrialAsset(AssetBaseModel):
    asset_id: UUID = Field(
        default_factory=uuid4
    )

    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_code: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_name: str = Field(
        min_length=1,
        max_length=200,
    )

    asset_type: AssetType

    description: str | None = None

    status: AssetStatus = AssetStatus.ACTIVE

    criticality: AssetCriticality = (
        AssetCriticality.MEDIUM
    )

    location: AssetLocation

    manufacturer: AssetManufacturer | None = None

    sensors: list[AssetSensor] = Field(
        default_factory=list
    )

    health: AssetHealth = Field(
        default_factory=AssetHealth
    )

    maintenance: AssetMaintenance = Field(
        default_factory=AssetMaintenance
    )

    warranty: AssetWarranty | None = None

    financials: AssetFinancials | None = None

    operations: AssetOperationalMetrics = Field(
        default_factory=AssetOperationalMetrics
    )

    installation_date: date | None = None

    commissioning_date: date | None = None

    expected_lifespan_years: float | None = Field(
        default=None,
        gt=0,
    )

    parent_asset_id: UUID | None = None

    tags: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned = [
            value.strip()
            for value in values
            if value.strip()
        ]

        return list(
            dict.fromkeys(cleaned)
        )

    @model_validator(mode="after")
    def validate_asset_dates(
        self,
    ) -> "IndustrialAsset":
        if (
            self.installation_date
            and self.commissioning_date
            and self.commissioning_date
            < self.installation_date
        ):
            raise ValueError(
                "Commissioning date cannot "
                "precede installation date"
            )

        if (
            self.parent_asset_id
            == self.asset_id
        ):
            raise ValueError(
                "Asset cannot be its own parent"
            )

        sensor_ids = [
            sensor.sensor_id
            for sensor in self.sensors
        ]

        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError(
                "Duplicate sensor IDs"
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )

    def update_health(
        self,
        *,
        health_score: float | None = None,
        anomaly_score: float | None = None,
        failure_probability: float | None = None,
        remaining_useful_life_hours: (
            float | None
        ) = None,
        visual_health_score: float | None = None,
        model_version: str | None = None,
    ) -> None:
        updates = {
            "health_score": health_score,
            "anomaly_score": anomaly_score,
            "failure_probability": failure_probability,
            "remaining_useful_life_hours": (
                remaining_useful_life_hours
            ),
            "visual_health_score": (
                visual_health_score
            ),
            "model_version": model_version,
        }

        # Validate the complete new state before
        # assigning it to the asset.
        current = self.health.model_dump()

        for key, value in updates.items():
            if value is not None:
                current[key] = value

        current["last_evaluated_at"] = (
            datetime.now(timezone.utc)
        )

        validated = AssetHealth.model_validate(
            current
        )

        self.health = validated

        self.updated_at = datetime.now(
            timezone.utc
        )

    def mark_maintenance(
        self,
        *,
        maintenance_date: date,
        next_maintenance_date: date | None = None,
        notes: str | None = None,
    ) -> None:
        if (
            next_maintenance_date is not None
            and next_maintenance_date
            < maintenance_date
        ):
            raise ValueError(
                "Next maintenance date cannot "
                "precede completed maintenance"
            )

        self.maintenance.last_maintenance_date = (
            maintenance_date
        )

        self.maintenance.next_maintenance_date = (
            next_maintenance_date
        )

        self.maintenance.total_maintenance_events += 1

        if notes is not None:
            self.maintenance.maintenance_notes = notes

        self.updated_at = datetime.now(
            timezone.utc
        )


# ============================================================
# CREATE / UPDATE / RESPONSE SCHEMAS
# ============================================================

class AssetCreate(AssetBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_code: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_name: str = Field(
        min_length=1,
        max_length=200,
    )

    asset_type: AssetType

    description: str | None = None

    criticality: AssetCriticality = (
        AssetCriticality.MEDIUM
    )

    location: AssetLocation

    manufacturer: AssetManufacturer | None = None

    sensors: list[AssetSensor] = Field(
        default_factory=list
    )

    installation_date: date | None = None

    tags: list[str] = Field(
        default_factory=list
    )


class AssetUpdate(AssetBaseModel):
    asset_name: str | None = None

    description: str | None = None

    status: AssetStatus | None = None

    criticality: AssetCriticality | None = None

    location: AssetLocation | None = None

    manufacturer: AssetManufacturer | None = None

    sensors: list[AssetSensor] | None = None

    tags: list[str] | None = None


class AssetResponse(IndustrialAsset):
    pass


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    asset = IndustrialAsset(
        company_id="company-001",
        asset_code="PUMP-101",
        asset_name="Main Cooling Water Pump",
        asset_type=AssetType.PUMP,
        criticality=AssetCriticality.HIGH,
        location=AssetLocation(
            site_id="SITE-01",
            site_name="Manufacturing Plant",
            building="Production Block A",
            production_line="LINE-01",
            zone="PUMP-ROOM",
        ),
        manufacturer=AssetManufacturer(
            manufacturer_name="Example Manufacturer",
            model_number="PUMP-X100",
            serial_number="SN-000101",
        ),
        sensors=[
            AssetSensor(
                sensor_id="TEMP-101",
                sensor_type=SensorType.TEMPERATURE,
                unit="C",
                warning_threshold=75.0,
                critical_threshold=95.0,
            ),
            AssetSensor(
                sensor_id="VIB-101",
                sensor_type=SensorType.VIBRATION,
                unit="mm/s",
                warning_threshold=5.0,
                critical_threshold=8.0,
            ),
        ],
        financials=AssetFinancials(
            currency="INR",
            purchase_cost=450000.0,
            replacement_cost=520000.0,
            estimated_downtime_cost_per_hour=15000.0,
        ),
    )

    asset.update_health(
        health_score=82.5,
        anomaly_score=0.18,
        failure_probability=0.06,
        remaining_useful_life_hours=4200.0,
        visual_health_score=88.0,
        model_version="v1.0",
    )

    print(
        json.dumps(
            asset.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()
