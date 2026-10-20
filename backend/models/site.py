"""
app/ai/ml/models/site.py

Industrial Site Intelligence Models
===================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant industrial site registry
- Factories, plants, warehouses and facilities
- Site addresses and geographic coordinates
- Buildings, floors and operational zones
- Production lines and equipment references
- IoT gateways and sensor infrastructure
- Site health and operational performance
- Safety infrastructure and emergency readiness
- Maintenance and inspection configuration
- Site lifecycle management
- Create, update, response and summary schemas

Domain model only.

Database persistence, authorization, real-time
telemetry ingestion, emergency dispatch and
physical equipment control belong in separate
services and certified industrial systems.
"""

from __future__ import annotations

from datetime import datetime, timezone
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
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_datetime(
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

class SiteType(str, Enum):
    FACTORY = "factory"
    MANUFACTURING_PLANT = "manufacturing_plant"
    POWER_PLANT = "power_plant"
    CHEMICAL_PLANT = "chemical_plant"
    REFINERY = "refinery"
    OIL_GAS_FACILITY = "oil_gas_facility"
    WAREHOUSE = "warehouse"
    DISTRIBUTION_CENTER = "distribution_center"
    DATA_CENTER = "data_center"
    MINING_SITE = "mining_site"
    WATER_TREATMENT = "water_treatment"
    FOOD_PROCESSING = "food_processing"
    PHARMACEUTICAL = "pharmaceutical"
    AUTOMOTIVE_PLANT = "automotive_plant"
    CONSTRUCTION_SITE = "construction_site"
    RESEARCH_FACILITY = "research_facility"
    OTHER = "other"


class SiteStatus(str, Enum):
    PLANNED = "planned"
    COMMISSIONING = "commissioning"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    DECOMMISSIONED = "decommissioned"


class SiteCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SiteHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class BuildingType(str, Enum):
    PRODUCTION = "production"
    WAREHOUSE = "warehouse"
    OFFICE = "office"
    CONTROL_ROOM = "control_room"
    UTILITY = "utility"
    LABORATORY = "laboratory"
    MAINTENANCE = "maintenance"
    STORAGE = "storage"
    OTHER = "other"


class ZoneType(str, Enum):
    PRODUCTION = "production"
    ASSEMBLY = "assembly"
    STORAGE = "storage"
    LOADING = "loading"
    ELECTRICAL = "electrical"
    HIGH_TEMPERATURE = "high_temperature"
    HIGH_PRESSURE = "high_pressure"
    CHEMICAL = "chemical"
    RESTRICTED = "restricted"
    HAZARDOUS = "hazardous"
    EMERGENCY = "emergency"
    SAFE_ASSEMBLY = "safe_assembly"
    GENERAL = "general"


class ZoneRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class GatewayStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class EmergencyReadinessStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    READY = "ready"
    NEEDS_ATTENTION = "needs_attention"
    NOT_READY = "not_ready"


class SiteContactRole(str, Enum):
    SITE_MANAGER = "site_manager"
    PLANT_MANAGER = "plant_manager"
    MAINTENANCE_MANAGER = "maintenance_manager"
    SAFETY_OFFICER = "safety_officer"
    EMERGENCY_COORDINATOR = "emergency_coordinator"
    SECURITY_OFFICER = "security_officer"
    OPERATIONS_MANAGER = "operations_manager"
    OTHER = "other"


# ============================================================
# BASE MODEL
# ============================================================

class SiteBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ============================================================
# ADDRESS
# ============================================================

class SiteAddress(SiteBaseModel):
    address_line_1: str = Field(min_length=1)
    address_line_2: str | None = None

    city: str = Field(min_length=1)
    state: str | None = None

    postal_code: str | None = None

    country: str = Field(min_length=1)
    country_code: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{2}$",
    )

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        allow_inf_nan=False,
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        allow_inf_nan=False,
    )

    timezone_name: str = "UTC"

    @model_validator(mode="after")
    def validate_coordinates(self) -> "SiteAddress":
        if (self.latitude is None) != (
            self.longitude is None
        ):
            raise ValueError(
                "Latitude and longitude must "
                "be provided together"
            )

        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                "Invalid IANA timezone"
            ) from exc

        return self


# ============================================================
# SITE CONTACT
# ============================================================

class SiteContact(SiteBaseModel):
    contact_id: UUID = Field(
        default_factory=uuid4
    )

    user_id: str | None = None

    full_name: str = Field(min_length=1)

    role: SiteContactRole

    email: str | None = None
    phone_number: str | None = None

    primary_contact: bool = False

    emergency_contact: bool = False

    active: bool = True


# ============================================================
# BUILDING
# ============================================================

class SiteBuilding(SiteBaseModel):
    building_id: str = Field(min_length=1)

    building_name: str = Field(min_length=1)

    building_type: BuildingType = (
        BuildingType.PRODUCTION
    )

    number_of_floors: int = Field(
        default=1,
        ge=1,
    )

    floor_labels: list[str] = Field(
        default_factory=list
    )

    total_area_sq_m: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    occupancy_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    active: bool = True

    @model_validator(mode="after")
    def validate_floors(self) -> "SiteBuilding":
        if len(self.floor_labels) != len(
            set(self.floor_labels)
        ):
            raise ValueError(
                "Duplicate floor labels"
            )

        if len(self.floor_labels) > self.number_of_floors:
            raise ValueError(
                "Floor labels exceed declared floor count"
            )

        return self


# ============================================================
# OPERATIONAL ZONE
# ============================================================

class SiteZone(SiteBaseModel):
    zone_id: str = Field(min_length=1)

    zone_name: str = Field(min_length=1)

    zone_type: ZoneType

    risk_level: ZoneRiskLevel = ZoneRiskLevel.LOW

    building_id: str | None = None
    floor_label: str | None = None

    restricted_access: bool = False

    authorized_role_ids: list[str] = Field(
        default_factory=list
    )

    maximum_occupancy: int | None = Field(
        default=None,
        ge=0,
    )

    camera_ids: list[str] = Field(
        default_factory=list
    )

    sensor_ids: list[str] = Field(
        default_factory=list
    )

    emergency_exit_ids: list[str] = Field(
        default_factory=list
    )

    active: bool = True

    description: str | None = None

    @model_validator(mode="after")
    def validate_zone(self) -> "SiteZone":
        if self.floor_label and not self.building_id:
            raise ValueError(
                "floor_label requires building_id"
            )

        if (
            self.zone_type == ZoneType.RESTRICTED.value
            and not self.restricted_access
        ):
            raise ValueError(
                "Restricted zones must enforce "
                "restricted_access"
            )

        return self


# ============================================================
# PRODUCTION LINE
# ============================================================

class SiteProductionLine(SiteBaseModel):
    production_line_id: str = Field(min_length=1)

    line_name: str = Field(min_length=1)

    building_id: str | None = None
    zone_id: str | None = None

    asset_ids: list[str] = Field(
        default_factory=list
    )

    machine_ids: list[str] = Field(
        default_factory=list
    )

    designed_capacity_per_hour: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    actual_output_per_hour: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    availability_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    performance_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    quality_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    active: bool = True

    @property
    def oee_percent(self) -> float | None:
        values = (
            self.availability_percent,
            self.performance_percent,
            self.quality_percent,
        )

        if any(value is None for value in values):
            return None

        return round(
            (
                self.availability_percent
                * self.performance_percent
                * self.quality_percent
            ) / 10000,
            2,
        )


# ============================================================
# IOT GATEWAY
# ============================================================

class SiteIoTGateway(SiteBaseModel):
    gateway_id: str = Field(min_length=1)

    gateway_name: str = Field(min_length=1)

    protocol: str = Field(min_length=1)

    status: GatewayStatus = GatewayStatus.UNKNOWN

    building_id: str | None = None
    zone_id: str | None = None

    sensor_ids: list[str] = Field(
        default_factory=list
    )

    firmware_version: str | None = None

    last_seen_at: datetime | None = None

    @field_validator("last_seen_at")
    @classmethod
    def validate_last_seen(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)


# ============================================================
# SITE SAFETY INFRASTRUCTURE
# ============================================================

class SiteSafetyInfrastructure(SiteBaseModel):
    fire_detection_enabled: bool = False
    smoke_detection_enabled: bool = False
    gas_detection_enabled: bool = False

    emergency_alarm_system_available: bool = False

    certified_fire_alarm_reference: str | None = None

    emergency_exit_count: int = Field(
        default=0,
        ge=0,
    )

    assembly_point_ids: list[str] = Field(
        default_factory=list
    )

    first_aid_station_count: int = Field(
        default=0,
        ge=0,
    )

    fire_extinguisher_count: int = Field(
        default=0,
        ge=0,
    )

    safety_officer_ids: list[str] = Field(
        default_factory=list
    )

    emergency_plan_reference: str | None = None

    last_safety_audit_at: datetime | None = None

    next_safety_audit_at: datetime | None = None

    @field_validator(
        "last_safety_audit_at",
        "next_safety_audit_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)


# ============================================================
# EMERGENCY READINESS
# ============================================================

class SiteEmergencyReadiness(SiteBaseModel):
    status: EmergencyReadinessStatus = (
        EmergencyReadinessStatus.NOT_ASSESSED
    )

    emergency_response_team_ids: list[str] = Field(
        default_factory=list
    )

    evacuation_plan_reference: str | None = None

    emergency_contact_reference: str | None = None

    last_drill_at: datetime | None = None
    next_drill_at: datetime | None = None

    assessed_by: str | None = None
    assessed_at: datetime | None = None

    outstanding_findings: list[str] = Field(
        default_factory=list
    )

    notes: str | None = None

    @field_validator(
        "last_drill_at",
        "next_drill_at",
        "assessed_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_readiness(
        self,
    ) -> "SiteEmergencyReadiness":
        if self.status != (
            EmergencyReadinessStatus.NOT_ASSESSED.value
        ):
            if not self.assessed_by or not self.assessed_at:
                raise ValueError(
                    "Readiness assessment requires "
                    "assessor and timestamp"
                )

        return self


# ============================================================
# MAINTENANCE SETTINGS
# ============================================================

class SiteMaintenanceSettings(SiteBaseModel):
    preventive_maintenance_enabled: bool = True

    predictive_maintenance_enabled: bool = True

    inspection_interval_days: int = Field(
        default=30,
        ge=1,
    )

    maintenance_planning_horizon_days: int = Field(
        default=90,
        ge=1,
    )

    maintenance_manager_ids: list[str] = Field(
        default_factory=list
    )

    work_order_approval_required: bool = True

    automated_equipment_control_enabled: bool = False

    @model_validator(mode="after")
    def enforce_safety(
        self,
    ) -> "SiteMaintenanceSettings":
        if self.automated_equipment_control_enabled:
            raise ValueError(
                "Automated physical equipment control "
                "is not supported by this AI application"
            )

        return self


# ============================================================
# SITE HEALTH
# ============================================================

class SiteHealth(SiteBaseModel):
    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    status: SiteHealthStatus = (
        SiteHealthStatus.UNKNOWN
    )

    monitored_asset_count: int = Field(
        default=0,
        ge=0,
    )

    healthy_asset_count: int = Field(
        default=0,
        ge=0,
    )

    degraded_asset_count: int = Field(
        default=0,
        ge=0,
    )

    critical_asset_count: int = Field(
        default=0,
        ge=0,
    )

    active_anomaly_count: int = Field(
        default=0,
        ge=0,
    )

    open_safety_event_count: int = Field(
        default=0,
        ge=0,
    )

    open_maintenance_count: int = Field(
        default=0,
        ge=0,
    )

    assessment_method: str | None = None

    assessed_at: datetime | None = None

    @field_validator("assessed_at")
    @classmethod
    def validate_assessed_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_counts(self) -> "SiteHealth":
        categorized = (
            self.healthy_asset_count
            + self.degraded_asset_count
            + self.critical_asset_count
        )

        if categorized > self.monitored_asset_count:
            raise ValueError(
                "Categorized assets exceed "
                "monitored asset count"
            )

        return self


# ============================================================
# SITE OPERATIONS
# ============================================================

class SiteOperationalMetrics(SiteBaseModel):
    availability_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    performance_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    quality_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    production_output: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    production_target: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    downtime_hours: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    energy_consumption_kwh: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    estimated_downtime_cost: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    period_start: datetime | None = None
    period_end: datetime | None = None

    @field_validator(
        "period_start",
        "period_end",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_period(
        self,
    ) -> "SiteOperationalMetrics":
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_end <= self.period_start
        ):
            raise ValueError(
                "period_end must follow period_start"
            )

        return self

    @property
    def oee_percent(self) -> float | None:
        if any(
            value is None
            for value in (
                self.availability_percent,
                self.performance_percent,
                self.quality_percent,
            )
        ):
            return None

        return round(
            (
                self.availability_percent
                * self.performance_percent
                * self.quality_percent
            ) / 10000,
            2,
        )


# ============================================================
# MAIN INDUSTRIAL SITE
# ============================================================

class IndustrialSite(SiteBaseModel):
    site_id: str = Field(min_length=1)

    site_uuid: UUID = Field(
        default_factory=uuid4
    )

    company_id: str = Field(min_length=1)

    site_code: str = Field(
        min_length=1,
        max_length=100,
    )

    site_name: str = Field(
        min_length=1,
        max_length=250,
    )

    site_type: SiteType

    status: SiteStatus = SiteStatus.PLANNED

    criticality: SiteCriticality = (
        SiteCriticality.MEDIUM
    )

    description: str | None = None

    address: SiteAddress

    contacts: list[SiteContact] = Field(
        default_factory=list
    )

    buildings: list[SiteBuilding] = Field(
        default_factory=list
    )

    zones: list[SiteZone] = Field(
        default_factory=list
    )

    production_lines: list[
        SiteProductionLine
    ] = Field(default_factory=list)

    gateways: list[SiteIoTGateway] = Field(
        default_factory=list
    )

    asset_ids: list[str] = Field(
        default_factory=list
    )

    machine_ids: list[str] = Field(
        default_factory=list
    )

    sensor_ids: list[str] = Field(
        default_factory=list
    )

    safety_infrastructure: (
        SiteSafetyInfrastructure
    ) = Field(
        default_factory=SiteSafetyInfrastructure
    )

    emergency_readiness: (
        SiteEmergencyReadiness
    ) = Field(
        default_factory=SiteEmergencyReadiness
    )

    maintenance_settings: (
        SiteMaintenanceSettings
    ) = Field(
        default_factory=SiteMaintenanceSettings
    )

    health: SiteHealth = Field(
        default_factory=SiteHealth
    )

    operational_metrics: (
        SiteOperationalMetrics
    ) = Field(
        default_factory=SiteOperationalMetrics
    )

    commissioned_at: datetime | None = None
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
        "commissioned_at",
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
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_site(
        self,
    ) -> "IndustrialSite":
        collections = (
            ("building", [
                item.building_id
                for item in self.buildings
            ]),
            ("zone", [
                item.zone_id
                for item in self.zones
            ]),
            ("production line", [
                item.production_line_id
                for item in self.production_lines
            ]),
            ("gateway", [
                item.gateway_id
                for item in self.gateways
            ]),
            ("asset", self.asset_ids),
            ("machine", self.machine_ids),
            ("sensor", self.sensor_ids),
        )

        for label, identifiers in collections:
            if len(identifiers) != len(
                set(identifiers)
            ):
                raise ValueError(
                    f"Duplicate {label} identifiers"
                )

        building_ids = {
            building.building_id
            for building in self.buildings
        }

        zone_ids = {
            zone.zone_id
            for zone in self.zones
        }

        for zone in self.zones:
            if (
                zone.building_id is not None
                and zone.building_id not in building_ids
            ):
                raise ValueError(
                    f"Unknown building for zone "
                    f"{zone.zone_id}"
                )

        for line in self.production_lines:
            if (
                line.building_id is not None
                and line.building_id not in building_ids
            ):
                raise ValueError(
                    "Production line references "
                    "unknown building"
                )

            if (
                line.zone_id is not None
                and line.zone_id not in zone_ids
            ):
                raise ValueError(
                    "Production line references "
                    "unknown zone"
                )

        for gateway in self.gateways:
            if (
                gateway.building_id is not None
                and gateway.building_id not in building_ids
            ):
                raise ValueError(
                    "Gateway references unknown building"
                )

            if (
                gateway.zone_id is not None
                and gateway.zone_id not in zone_ids
            ):
                raise ValueError(
                    "Gateway references unknown zone"
                )

        return self

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = utc_now()

    # --------------------------------------------------------
    # SITE LIFECYCLE
    # --------------------------------------------------------

    def activate(self) -> None:
        if self.status == SiteStatus.DECOMMISSIONED.value:
            raise ValueError(
                "Decommissioned site cannot be activated"
            )

        if self.status == SiteStatus.ACTIVE.value:
            raise ValueError(
                "Site is already active"
            )

        self.status = SiteStatus.ACTIVE
        self.activated_at = utc_now()
        self._touch()

    def suspend(self) -> None:
        if self.status != SiteStatus.ACTIVE.value:
            raise ValueError(
                "Only active sites can be suspended"
            )

        self.status = SiteStatus.SUSPENDED
        self._touch()

    def decommission(self) -> None:
        if self.status == SiteStatus.DECOMMISSIONED.value:
            raise ValueError(
                "Site already decommissioned"
            )

        self.status = SiteStatus.DECOMMISSIONED
        self.decommissioned_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------------

    def add_building(
        self,
        building: SiteBuilding,
    ) -> None:
        if any(
            item.building_id == building.building_id
            for item in self.buildings
        ):
            raise ValueError(
                "Building already exists"
            )

        self.buildings.append(building)
        self._touch()

    def get_building(
        self,
        building_id: str,
    ) -> SiteBuilding | None:
        return next(
            (
                building
                for building in self.buildings
                if building.building_id == building_id
            ),
            None,
        )

    # --------------------------------------------------------
    # ZONES
    # --------------------------------------------------------

    def add_zone(
        self,
        zone: SiteZone,
    ) -> None:
        if any(
            item.zone_id == zone.zone_id
            for item in self.zones
        ):
            raise ValueError(
                "Zone already exists"
            )

        if (
            zone.building_id is not None
            and self.get_building(zone.building_id) is None
        ):
            raise ValueError(
                "Zone references unknown building"
            )

        self.zones.append(zone)
        self._touch()

    def get_zone(
        self,
        zone_id: str,
    ) -> SiteZone | None:
        return next(
            (
                zone
                for zone in self.zones
                if zone.zone_id == zone_id
            ),
            None,
        )

    # --------------------------------------------------------
    # PRODUCTION LINES
    # --------------------------------------------------------

    def add_production_line(
        self,
        line: SiteProductionLine,
    ) -> None:
        if any(
            item.production_line_id
            == line.production_line_id
            for item in self.production_lines
        ):
            raise ValueError(
                "Production line already exists"
            )

        if (
            line.building_id is not None
            and self.get_building(line.building_id) is None
        ):
            raise ValueError(
                "Production line references "
                "unknown building"
            )

        if (
            line.zone_id is not None
            and self.get_zone(line.zone_id) is None
        ):
            raise ValueError(
                "Production line references unknown zone"
            )

        self.production_lines.append(line)
        self._touch()

    # --------------------------------------------------------
    # GATEWAYS
    # --------------------------------------------------------

    def add_gateway(
        self,
        gateway: SiteIoTGateway,
    ) -> None:
        if any(
            item.gateway_id == gateway.gateway_id
            for item in self.gateways
        ):
            raise ValueError(
                "Gateway already exists"
            )

        if (
            gateway.building_id is not None
            and self.get_building(
                gateway.building_id
            ) is None
        ):
            raise ValueError(
                "Gateway references unknown building"
            )

        if (
            gateway.zone_id is not None
            and self.get_zone(gateway.zone_id) is None
        ):
            raise ValueError(
                "Gateway references unknown zone"
            )

        self.gateways.append(gateway)
        self._touch()

    # --------------------------------------------------------
    # ASSET REGISTRY
    # --------------------------------------------------------

    def register_asset(
        self,
        asset_id: str,
    ) -> None:
        if not asset_id.strip():
            raise ValueError(
                "asset_id is required"
            )

        if asset_id in self.asset_ids:
            raise ValueError(
                "Asset already registered"
            )

        self.asset_ids.append(asset_id)
        self._touch()

    def register_machine(
        self,
        machine_id: str,
    ) -> None:
        if not machine_id.strip():
            raise ValueError(
                "machine_id is required"
            )

        if machine_id in self.machine_ids:
            raise ValueError(
                "Machine already registered"
            )

        self.machine_ids.append(machine_id)
        self._touch()

    def register_sensor(
        self,
        sensor_id: str,
    ) -> None:
        if not sensor_id.strip():
            raise ValueError(
                "sensor_id is required"
            )

        if sensor_id in self.sensor_ids:
            raise ValueError(
                "Sensor already registered"
            )

        self.sensor_ids.append(sensor_id)
        self._touch()

    # --------------------------------------------------------
    # UPDATE HEALTH
    # --------------------------------------------------------

    def update_health(
        self,
        health: SiteHealth,
    ) -> None:
        self.health = health
        self._touch()

    # --------------------------------------------------------
    # UPDATE OPERATIONS
    # --------------------------------------------------------

    def update_operational_metrics(
        self,
        metrics: SiteOperationalMetrics,
    ) -> None:
        self.operational_metrics = metrics
        self._touch()

    # --------------------------------------------------------
    # SITE ANALYTICS
    # --------------------------------------------------------

    @property
    def building_count(self) -> int:
        return len(self.buildings)

    @property
    def zone_count(self) -> int:
        return len(self.zones)

    @property
    def production_line_count(self) -> int:
        return len(self.production_lines)

    @property
    def asset_count(self) -> int:
        return len(self.asset_ids)

    @property
    def sensor_count(self) -> int:
        return len(self.sensor_ids)

    @property
    def online_gateway_count(self) -> int:
        return sum(
            gateway.status == GatewayStatus.ONLINE.value
            for gateway in self.gateways
        )

    @property
    def high_risk_zone_count(self) -> int:
        return sum(
            zone.risk_level in {
                ZoneRiskLevel.HIGH.value,
                ZoneRiskLevel.CRITICAL.value,
            }
            for zone in self.zones
        )

    def requires_engineering_review(self) -> bool:
        return (
            self.health.status
            in {
                SiteHealthStatus.DEGRADED.value,
                SiteHealthStatus.CRITICAL.value,
            }
            or self.health.critical_asset_count > 0
            or self.health.open_safety_event_count > 0
            or self.emergency_readiness.status
            == EmergencyReadinessStatus.NOT_READY.value
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class SiteCreate(SiteBaseModel):
    site_id: str = Field(min_length=1)

    company_id: str = Field(min_length=1)

    site_code: str = Field(min_length=1)

    site_name: str = Field(min_length=1)

    site_type: SiteType

    address: SiteAddress

    description: str | None = None

    criticality: SiteCriticality = (
        SiteCriticality.MEDIUM
    )

    contacts: list[SiteContact] = Field(
        default_factory=list
    )


# ============================================================
# UPDATE SCHEMA
# ============================================================

class SiteUpdate(SiteBaseModel):
    site_name: str | None = None

    description: str | None = None

    address: SiteAddress | None = None

    criticality: SiteCriticality | None = None

    maintenance_settings: (
        SiteMaintenanceSettings | None
    ) = None

    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class SiteResponse(IndustrialSite):
    pass


# ============================================================
# SUMMARY SCHEMA
# ============================================================

class SiteSummary(SiteBaseModel):
    site_id: str

    company_id: str

    site_code: str
    site_name: str

    site_type: SiteType
    status: SiteStatus

    criticality: SiteCriticality

    city: str
    country: str

    building_count: int
    zone_count: int
    production_line_count: int

    asset_count: int
    machine_count: int
    sensor_count: int

    online_gateway_count: int
    high_risk_zone_count: int

    health_score: float | None
    health_status: SiteHealthStatus

    open_safety_event_count: int
    critical_asset_count: int

    oee_percent: float | None

    emergency_readiness_status: (
        EmergencyReadinessStatus
    )

    requires_engineering_review: bool

    updated_at: datetime

    @classmethod
    def from_site(
        cls,
        site: IndustrialSite,
    ) -> "SiteSummary":
        return cls(
            site_id=site.site_id,
            company_id=site.company_id,
            site_code=site.site_code,
            site_name=site.site_name,
            site_type=site.site_type,
            status=site.status,
            criticality=site.criticality,
            city=site.address.city,
            country=site.address.country,
            building_count=site.building_count,
            zone_count=site.zone_count,
            production_line_count=(
                site.production_line_count
            ),
            asset_count=site.asset_count,
            machine_count=len(site.machine_ids),
            sensor_count=site.sensor_count,
            online_gateway_count=(
                site.online_gateway_count
            ),
            high_risk_zone_count=(
                site.high_risk_zone_count
            ),
            health_score=site.health.health_score,
            health_status=site.health.status,
            open_safety_event_count=(
                site.health.open_safety_event_count
            ),
            critical_asset_count=(
                site.health.critical_asset_count
            ),
            oee_percent=(
                site.operational_metrics.oee_percent
            ),
            emergency_readiness_status=(
                site.emergency_readiness.status
            ),
            requires_engineering_review=(
                site.requires_engineering_review()
            ),
            updated_at=site.updated_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    site = IndustrialSite(
        site_id="SITE-001",
        company_id="COMPANY-001",
        site_code="PLANT-KL-01",
        site_name="Industrial Manufacturing Plant",
        site_type=SiteType.MANUFACTURING_PLANT,
        status=SiteStatus.ACTIVE,
        criticality=SiteCriticality.HIGH,
        address=SiteAddress(
            address_line_1="Industrial Estate",
            city="Kochi",
            state="Kerala",
            country="India",
            country_code="IN",
            timezone_name="Asia/Kolkata",
        ),
        contacts=[
            SiteContact(
                full_name="Plant Operations Manager",
                role=SiteContactRole.PLANT_MANAGER,
                primary_contact=True,
            )
        ],
    )

    site.add_building(
        SiteBuilding(
            building_id="BLDG-001",
            building_name="Production Building",
            building_type=BuildingType.PRODUCTION,
            number_of_floors=2,
            floor_labels=[
                "Ground",
                "First",
            ],
        )
    )

    site.add_zone(
        SiteZone(
            zone_id="ZONE-001",
            zone_name="Motor Assembly Area",
            zone_type=ZoneType.ASSEMBLY,
            risk_level=ZoneRiskLevel.MODERATE,
            building_id="BLDG-001",
            floor_label="Ground",
        )
    )

    site.add_production_line(
        SiteProductionLine(
            production_line_id="LINE-001",
            line_name="Motor Assembly Line",
            building_id="BLDG-001",
            zone_id="ZONE-001",
            designed_capacity_per_hour=100,
            actual_output_per_hour=82,
            availability_percent=92,
            performance_percent=89,
            quality_percent=98,
        )
    )

    site.register_asset("ASSET-101")
    site.register_machine("MACHINE-101")
    site.register_sensor("TEMP-101")
    site.register_sensor("VIB-101")

    site.update_health(
        SiteHealth(
            health_score=87.5,
            status=SiteHealthStatus.HEALTHY,
            monitored_asset_count=1,
            healthy_asset_count=1,
            assessed_at=utc_now(),
        )
    )

    site.update_operational_metrics(
        SiteOperationalMetrics(
            availability_percent=92,
            performance_percent=89,
            quality_percent=98,
            production_output=820,
            production_target=1000,
            downtime_hours=4.5,
            energy_consumption_kwh=1500,
        )
    )

    summary = SiteSummary.from_site(site)

    print(
        json.dumps(
            summary.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()