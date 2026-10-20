"""
app/ai/ml/models/component.py

Industrial Equipment Component Models
=====================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Industrial component registry
- Parent asset and equipment relationships
- Component hierarchy
- Component condition and health
- Anomaly, failure and RUL predictions
- Component-level sensor measurements
- Visual inspection findings
- Maintenance history
- Repair and replacement intelligence
- Financial and lifecycle information
- Create, update and response schemas
- Tenant-aware identifiers

NOTE:
    These are validated domain models,
    not trainable ML estimators or ORM models.

    ML scores and probabilities must originate
    from validated inference pipelines.
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

class ComponentType(str, Enum):
    BEARING = "bearing"
    GEAR = "gear"
    GEARBOX = "gearbox"
    SHAFT = "shaft"
    MOTOR = "motor"
    ROTOR = "rotor"
    STATOR = "stator"
    IMPELLER = "impeller"
    PUMP_SEAL = "pump_seal"
    MECHANICAL_SEAL = "mechanical_seal"
    VALVE = "valve"
    ACTUATOR = "actuator"
    COUPLING = "coupling"
    BELT = "belt"
    PULLEY = "pulley"
    CHAIN = "chain"
    FILTER = "filter"
    FAN = "fan"
    COMPRESSOR_STAGE = "compressor_stage"
    HYDRAULIC_CYLINDER = "hydraulic_cylinder"
    HYDRAULIC_PUMP = "hydraulic_pump"
    PIPE = "pipe"
    FLANGE = "flange"
    GASKET = "gasket"
    HEAT_EXCHANGER = "heat_exchanger"
    SENSOR = "sensor"
    CONTROLLER = "controller"
    CIRCUIT_BREAKER = "circuit_breaker"
    TRANSFORMER_WINDING = "transformer_winding"
    BATTERY = "battery"
    POWER_SUPPLY = "power_supply"
    OTHER = "other"


class ComponentStatus(str, Enum):
    INSTALLED = "installed"
    ACTIVE = "active"
    IDLE = "idle"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNDER_INSPECTION = "under_inspection"
    UNDER_REPAIR = "under_repair"
    REMOVED = "removed"
    REPLACED = "replaced"
    DECOMMISSIONED = "decommissioned"


class ComponentCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentFailureMode(str, Enum):
    WEAR = "wear"
    FATIGUE = "fatigue"
    CORROSION = "corrosion"
    CRACK = "crack"
    OVERHEATING = "overheating"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"
    LUBRICATION_FAILURE = "lubrication_failure"
    LEAKAGE = "leakage"
    ELECTRICAL_FAULT = "electrical_fault"
    INSULATION_FAILURE = "insulation_failure"
    VIBRATION_DAMAGE = "vibration_damage"
    BLOCKAGE = "blockage"
    CONTAMINATION = "contamination"
    STRUCTURAL_DAMAGE = "structural_damage"
    UNKNOWN = "unknown"


class InspectionType(str, Enum):
    VISUAL = "visual"
    THERMAL = "thermal"
    VIBRATION = "vibration"
    ULTRASONIC = "ultrasonic"
    ELECTRICAL = "electrical"
    PRESSURE = "pressure"
    LUBRICATION = "lubrication"
    MANUAL = "manual"
    AI_VISION = "ai_vision"


class InspectionSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class MaintenanceAction(str, Enum):
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    LUBRICATION = "lubrication"
    CALIBRATION = "calibration"
    ADJUSTMENT = "adjustment"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    OVERHAUL = "overhaul"
    TESTING = "testing"


class MaintenanceDecision(str, Enum):
    MONITOR = "monitor"
    INSPECT = "inspect"
    REPAIR = "repair"
    REPLACE = "replace"
    ENGINEERING_REVIEW = "engineering_review"


# ============================================================
# BASE MODEL
# ============================================================

class ComponentBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# MANUFACTURER
# ============================================================

class ComponentManufacturer(ComponentBaseModel):
    manufacturer_name: str = Field(
        min_length=1,
        max_length=200,
    )

    model_number: str | None = None
    part_number: str | None = None
    serial_number: str | None = None

    supplier_name: str | None = None
    supplier_reference: str | None = None

    country_of_origin: str | None = None


# ============================================================
# COMPONENT LOCATION
# ============================================================

class ComponentLocation(ComponentBaseModel):
    site_id: str | None = None
    production_line: str | None = None

    machine_section: str | None = None
    assembly_name: str | None = None

    mounting_position: str | None = None
    installation_notes: str | None = None


# ============================================================
# SENSOR MEASUREMENT
# ============================================================

class ComponentSensorReading(ComponentBaseModel):
    sensor_id: str = Field(
        min_length=1,
        max_length=100,
    )

    sensor_type: str = Field(
        min_length=1,
        max_length=100,
    )

    value: float

    unit: str = Field(
        min_length=1,
        max_length=30,
    )

    measured_at: datetime

    quality_flag: str = "valid"

    @field_validator("measured_at")
    @classmethod
    def validate_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "measured_at must be timezone-aware"
            )

        return value


# ============================================================
# COMPONENT HEALTH
# ============================================================

class ComponentHealth(ComponentBaseModel):
    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    health_status: ComponentHealthStatus = (
        ComponentHealthStatus.UNKNOWN
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

    remaining_useful_life_hours: float | None = Field(
        default=None,
        ge=0,
    )

    visual_health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    predicted_failure_mode: (
        ComponentFailureMode | None
    ) = None

    latest_prediction_id: str | None = None
    model_version: str | None = None

    last_evaluated_at: datetime | None = None


# ============================================================
# VISUAL INSPECTION
# ============================================================

class ComponentInspection(ComponentBaseModel):
    inspection_id: UUID = Field(
        default_factory=uuid4
    )

    inspection_type: InspectionType

    severity: InspectionSeverity = (
        InspectionSeverity.NONE
    )

    finding_type: str | None = None

    description: str | None = None

    affected_area_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    finding_count: int = Field(
        default=0,
        ge=0,
    )

    image_reference: str | None = None

    model_prediction_id: str | None = None

    requires_human_review: bool = True

    inspected_by: str | None = None

    inspected_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


# ============================================================
# MAINTENANCE RECORD
# ============================================================

class ComponentMaintenanceRecord(ComponentBaseModel):
    maintenance_id: UUID = Field(
        default_factory=uuid4
    )

    action: MaintenanceAction

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    performed_by: str | None = None

    work_order_id: str | None = None

    completed_at: datetime

    downtime_hours: float = Field(
        default=0,
        ge=0,
    )

    labor_hours: float = Field(
        default=0,
        ge=0,
    )

    parts_cost: float = Field(
        default=0,
        ge=0,
    )

    labor_cost: float = Field(
        default=0,
        ge=0,
    )

    additional_cost: float = Field(
        default=0,
        ge=0,
    )

    currency: str = "INR"

    notes: str | None = None

    @property
    def total_cost(self) -> float:
        return round(
            self.parts_cost
            + self.labor_cost
            + self.additional_cost,
            2,
        )


# ============================================================
# COMPONENT FINANCIALS
# ============================================================

class ComponentFinancials(ComponentBaseModel):
    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    purchase_cost: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_repair_cost: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_replacement_cost: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_current_value: float | None = Field(
        default=None,
        ge=0,
    )

    downtime_cost_per_hour: float | None = Field(
        default=None,
        ge=0,
    )

    expected_failure_cost: float | None = Field(
        default=None,
        ge=0,
    )

    annual_maintenance_cost: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# COMPONENT LIFECYCLE
# ============================================================

class ComponentLifecycle(ComponentBaseModel):
    installation_date: date | None = None

    commissioning_date: date | None = None

    warranty_end_date: date | None = None

    expected_life_hours: float | None = Field(
        default=None,
        gt=0,
    )

    operating_hours: float = Field(
        default=0,
        ge=0,
    )

    operating_cycles: int = Field(
        default=0,
        ge=0,
    )

    last_maintenance_date: date | None = None

    next_maintenance_date: date | None = None

    replacement_date: date | None = None

    @model_validator(mode="after")
    def validate_dates(
        self,
    ) -> "ComponentLifecycle":
        if (
            self.installation_date
            and self.commissioning_date
            and self.commissioning_date
            < self.installation_date
        ):
            raise ValueError(
                "Commissioning date cannot precede "
                "installation date"
            )

        if (
            self.last_maintenance_date
            and self.next_maintenance_date
            and self.next_maintenance_date
            < self.last_maintenance_date
        ):
            raise ValueError(
                "Next maintenance cannot precede "
                "last maintenance"
            )

        return self


# ============================================================
# REPAIR / REPLACEMENT ASSESSMENT
# ============================================================

class ComponentRepairAssessment(ComponentBaseModel):
    assessment_id: UUID = Field(
        default_factory=uuid4
    )

    recommended_decision: MaintenanceDecision = (
        MaintenanceDecision.ENGINEERING_REVIEW
    )

    repair_cost: float | None = Field(
        default=None,
        ge=0,
    )

    replacement_cost: float | None = Field(
        default=None,
        ge=0,
    )

    repair_downtime_hours: float | None = Field(
        default=None,
        ge=0,
    )

    replacement_downtime_hours: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_savings: float | None = None

    reasoning: str | None = None

    requires_approval: bool = True

    approved_by: str | None = None

    assessed_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


# ============================================================
# MAIN COMPONENT MODEL
# ============================================================

class IndustrialComponent(ComponentBaseModel):
    component_id: UUID = Field(
        default_factory=uuid4
    )

    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str = Field(
        min_length=1,
        max_length=100,
    )

    equipment_id: str | None = None

    parent_component_id: UUID | None = None

    component_code: str = Field(
        min_length=1,
        max_length=100,
    )

    component_name: str = Field(
        min_length=1,
        max_length=200,
    )

    component_type: ComponentType

    description: str | None = None

    status: ComponentStatus = (
        ComponentStatus.INSTALLED
    )

    criticality: ComponentCriticality = (
        ComponentCriticality.MEDIUM
    )

    manufacturer: (
        ComponentManufacturer | None
    ) = None

    location: (
        ComponentLocation | None
    ) = None

    health: ComponentHealth = Field(
        default_factory=ComponentHealth
    )

    lifecycle: ComponentLifecycle = Field(
        default_factory=ComponentLifecycle
    )

    financials: ComponentFinancials | None = None

    latest_sensor_readings: list[
        ComponentSensorReading
    ] = Field(
        default_factory=list
    )

    inspections: list[
        ComponentInspection
    ] = Field(
        default_factory=list
    )

    maintenance_history: list[
        ComponentMaintenanceRecord
    ] = Field(
        default_factory=list
    )

    repair_assessment: (
        ComponentRepairAssessment | None
    ) = None

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
    def normalize_tags(
        cls,
        values: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                value.strip()
                for value in values
                if value.strip()
            )
        )

    @model_validator(mode="after")
    def validate_component(
        self,
    ) -> "IndustrialComponent":
        if (
            self.parent_component_id
            == self.component_id
        ):
            raise ValueError(
                "Component cannot be its own parent"
            )

        return self

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )

    # --------------------------------------------------------
    # UPDATE HEALTH
    # --------------------------------------------------------

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
        predicted_failure_mode: (
            ComponentFailureMode | None
        ) = None,
        model_version: str | None = None,
    ) -> None:
        updates = {
            "health_score": health_score,
            "anomaly_score": anomaly_score,
            "failure_probability": failure_probability,
            "remaining_useful_life_hours": (
                remaining_useful_life_hours
            ),
            "visual_health_score": visual_health_score,
            "predicted_failure_mode": (
                predicted_failure_mode
            ),
            "model_version": model_version,
        }

        data = self.health.model_dump()

        for key, value in updates.items():
            if value is not None:
                data[key] = value

        data["last_evaluated_at"] = (
            datetime.now(timezone.utc)
        )

        self.health = ComponentHealth.model_validate(
            data
        )

        self.updated_at = datetime.now(
            timezone.utc
        )

    # --------------------------------------------------------
    # ADD INSPECTION
    # --------------------------------------------------------

    def add_inspection(
        self,
        inspection: ComponentInspection,
    ) -> None:
        if any(
            item.inspection_id == inspection.inspection_id
            for item in self.inspections
        ):
            raise ValueError(
                "Duplicate inspection ID"
            )

        self.inspections.append(
            inspection
        )

        self.updated_at = datetime.now(
            timezone.utc
        )

    # --------------------------------------------------------
    # ADD SENSOR READING
    # --------------------------------------------------------

    def add_sensor_reading(
        self,
        reading: ComponentSensorReading,
    ) -> None:
        # Maintain the latest reading per sensor.
        self.latest_sensor_readings = [
            existing
            for existing in self.latest_sensor_readings
            if existing.sensor_id != reading.sensor_id
        ]

        self.latest_sensor_readings.append(
            reading
        )

        self.updated_at = datetime.now(
            timezone.utc
        )

    # --------------------------------------------------------
    # ADD MAINTENANCE
    # --------------------------------------------------------

    def add_maintenance(
        self,
        record: ComponentMaintenanceRecord,
    ) -> None:
        if any(
            item.maintenance_id == record.maintenance_id
            for item in self.maintenance_history
        ):
            raise ValueError(
                "Duplicate maintenance ID"
            )

        self.maintenance_history.append(
            record
        )

        self.lifecycle.last_maintenance_date = (
            record.completed_at.date()
        )

        self.updated_at = datetime.now(
            timezone.utc
        )

    # --------------------------------------------------------
    # TOTAL MAINTENANCE COST
    # --------------------------------------------------------

    def total_maintenance_cost(self) -> float:
        return round(
            sum(
                record.total_cost
                for record in self.maintenance_history
            ),
            2,
        )

    # --------------------------------------------------------
    # COMPONENT AGE
    # --------------------------------------------------------

    def component_age_days(
        self,
        as_of: date | None = None,
    ) -> int | None:
        installed = self.lifecycle.installation_date

        if installed is None:
            return None

        reference_date = as_of or date.today()

        return max(
            0,
            (reference_date - installed).days,
        )

    # --------------------------------------------------------
    # FAILURE RISK INDICATOR
    # --------------------------------------------------------

    def needs_engineering_review(self) -> bool:
        if self.status in {
            ComponentStatus.FAILED.value,
            ComponentStatus.DEGRADED.value,
        }:
            return True

        if self.health.health_status in {
            ComponentHealthStatus.CRITICAL.value,
            ComponentHealthStatus.DEGRADED.value,
        }:
            return True

        return any(
            inspection.severity in {
                InspectionSeverity.HIGH.value,
                InspectionSeverity.CRITICAL.value,
            }
            for inspection in self.inspections
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class ComponentCreate(ComponentBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str = Field(
        min_length=1,
        max_length=100,
    )

    equipment_id: str | None = None

    parent_component_id: UUID | None = None

    component_code: str = Field(
        min_length=1,
        max_length=100,
    )

    component_name: str = Field(
        min_length=1,
        max_length=200,
    )

    component_type: ComponentType

    criticality: ComponentCriticality = (
        ComponentCriticality.MEDIUM
    )

    description: str | None = None

    manufacturer: (
        ComponentManufacturer | None
    ) = None

    location: ComponentLocation | None = None

    lifecycle: ComponentLifecycle = Field(
        default_factory=ComponentLifecycle
    )

    tags: list[str] = Field(
        default_factory=list
    )


# ============================================================
# UPDATE SCHEMA
# ============================================================

class ComponentUpdate(ComponentBaseModel):
    component_name: str | None = None

    description: str | None = None

    status: ComponentStatus | None = None

    criticality: ComponentCriticality | None = None

    manufacturer: (
        ComponentManufacturer | None
    ) = None

    location: ComponentLocation | None = None

    tags: list[str] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class ComponentResponse(IndustrialComponent):
    pass


# ============================================================
# COMPONENT SUMMARY
# ============================================================

class ComponentSummary(ComponentBaseModel):
    component_id: UUID
    company_id: str
    asset_id: str

    component_code: str
    component_name: str

    component_type: ComponentType

    status: ComponentStatus
    criticality: ComponentCriticality

    health_score: float | None
    health_status: ComponentHealthStatus

    anomaly_score: float | None
    failure_probability: float | None
    remaining_useful_life_hours: float | None

    inspection_count: int
    maintenance_count: int

    requires_engineering_review: bool

    @classmethod
    def from_component(
        cls,
        component: IndustrialComponent,
    ) -> "ComponentSummary":
        return cls(
            component_id=component.component_id,
            company_id=component.company_id,
            asset_id=component.asset_id,
            component_code=component.component_code,
            component_name=component.component_name,
            component_type=component.component_type,
            status=component.status,
            criticality=component.criticality,
            health_score=component.health.health_score,
            health_status=component.health.health_status,
            anomaly_score=component.health.anomaly_score,
            failure_probability=(
                component.health.failure_probability
            ),
            remaining_useful_life_hours=(
                component.health.remaining_useful_life_hours
            ),
            inspection_count=len(
                component.inspections
            ),
            maintenance_count=len(
                component.maintenance_history
            ),
            requires_engineering_review=(
                component.needs_engineering_review()
            ),
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    component = IndustrialComponent(
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="PUMP-101",
        component_code="BRG-101",
        component_name="Pump Drive-End Bearing",
        component_type=ComponentType.BEARING,
        status=ComponentStatus.ACTIVE,
        criticality=ComponentCriticality.HIGH,
        manufacturer=ComponentManufacturer(
            manufacturer_name="Example Bearings Ltd",
            model_number="BRG-6205",
            part_number="6205-2RS",
        ),
        location=ComponentLocation(
            site_id="SITE-01",
            machine_section="Drive End",
            assembly_name="Pump Motor Assembly",
        ),
        lifecycle=ComponentLifecycle(
            installation_date=date(2025, 1, 10),
            expected_life_hours=20000,
            operating_hours=6500,
        ),
        financials=ComponentFinancials(
            currency="INR",
            purchase_cost=8500,
            estimated_repair_cost=3500,
            estimated_replacement_cost=11000,
            downtime_cost_per_hour=15000,
        ),
    )

    component.add_sensor_reading(
        ComponentSensorReading(
            sensor_id="VIB-101",
            sensor_type="vibration",
            value=6.8,
            unit="mm/s",
            measured_at=datetime.now(
                timezone.utc
            ),
        )
    )

    component.update_health(
        health_score=68.5,
        anomaly_score=0.72,
        failure_probability=0.18,
        remaining_useful_life_hours=950,
        visual_health_score=82,
        predicted_failure_mode=(
            ComponentFailureMode.WEAR
        ),
        model_version="v1.0",
    )

    component.add_inspection(
        ComponentInspection(
            inspection_type=InspectionType.AI_VISION,
            severity=InspectionSeverity.MODERATE,
            finding_type="surface_wear",
            description=(
                "Visual wear candidate detected"
            ),
            requires_human_review=True,
        )
    )

    summary = ComponentSummary.from_component(
        component
    )

    print(
        json.dumps(
            summary.model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()