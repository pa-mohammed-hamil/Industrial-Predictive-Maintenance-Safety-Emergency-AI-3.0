"""
app/ai/ml/models/zone.py

Industrial Zone & Safety Intelligence Models
============================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant industrial zone registry
- Factory, building and floor relationships
- Production and restricted-access zones
- Hazardous-area classification
- Worker occupancy monitoring
- PPE requirements
- Camera, sensor and IoT gateway references
- Polygon geofencing
- Safety events and inspection references
- Emergency exits and assembly points
- Zone health and risk assessments
- Human-reviewed safety workflows
- Zone lifecycle management
- Create, update, response and summary schemas

Important:
    This module stores application-level records.
    It does not operate physical access controls,
    industrial interlocks or emergency systems.
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

class ZoneType(str, Enum):
    PRODUCTION = "production"
    ASSEMBLY = "assembly"
    MANUFACTURING = "manufacturing"
    MAINTENANCE = "maintenance"
    WAREHOUSE = "warehouse"
    STORAGE = "storage"
    LOADING = "loading"
    UNLOADING = "unloading"
    CONTROL_ROOM = "control_room"
    ELECTRICAL = "electrical"
    HIGH_VOLTAGE = "high_voltage"
    HIGH_TEMPERATURE = "high_temperature"
    HIGH_PRESSURE = "high_pressure"
    CHEMICAL = "chemical"
    GAS_STORAGE = "gas_storage"
    FLAMMABLE = "flammable"
    CONFINED_SPACE = "confined_space"
    ROBOTICS = "robotics"
    HEAVY_MACHINERY = "heavy_machinery"
    RESTRICTED = "restricted"
    HAZARDOUS = "hazardous"
    EMERGENCY = "emergency"
    EVACUATION_ROUTE = "evacuation_route"
    ASSEMBLY_POINT = "assembly_point"
    OFFICE = "office"
    GENERAL = "general"
    OTHER = "other"


class ZoneStatus(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    RESTRICTED = "restricted"
    UNDER_MAINTENANCE = "under_maintenance"
    TEMPORARILY_CLOSED = "temporarily_closed"
    INACTIVE = "inactive"
    DECOMMISSIONED = "decommissioned"


class ZoneRiskLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ZoneHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ZoneAccessLevel(str, Enum):
    PUBLIC = "public"
    EMPLOYEE = "employee"
    AUTHORIZED_PERSONNEL = "authorized_personnel"
    SUPERVISOR = "supervisor"
    SAFETY_OFFICER = "safety_officer"
    RESTRICTED = "restricted"


class ZoneHazardType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    GAS_LEAK = "gas_leak"
    CHEMICAL_EXPOSURE = "chemical_exposure"
    ELECTRICAL = "electrical"
    HIGH_VOLTAGE = "high_voltage"
    OVERHEATING = "overheating"
    HIGH_PRESSURE = "high_pressure"
    MOVING_MACHINERY = "moving_machinery"
    FALL = "fall"
    COLLISION = "collision"
    ROBOTIC_MOTION = "robotic_motion"
    NOISE = "noise"
    DUST = "dust"
    RADIATION = "radiation"
    CONFINED_SPACE = "confined_space"
    OTHER = "other"


class ZonePPEType(str, Enum):
    HELMET = "helmet"
    SAFETY_GLASSES = "safety_glasses"
    FACE_SHIELD = "face_shield"
    GLOVES = "gloves"
    SAFETY_SHOES = "safety_shoes"
    HIGH_VISIBILITY_VEST = "high_visibility_vest"
    RESPIRATOR = "respirator"
    HEARING_PROTECTION = "hearing_protection"
    PROTECTIVE_SUIT = "protective_suit"
    FALL_PROTECTION = "fall_protection"
    OTHER = "other"


class ZoneMonitoringStatus(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class ZoneOccupancyStatus(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    NEAR_CAPACITY = "near_capacity"
    OVER_CAPACITY = "over_capacity"


class ZoneReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================
# BASE MODEL
# ============================================================

class ZoneBaseModel(BaseModel):
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
# ZONE CONTEXT
# ============================================================

class ZoneContext(ZoneBaseModel):
    company_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)

    building_id: str | None = None
    floor_id: str | None = None

    production_line_id: str | None = None

    parent_zone_id: str | None = None


# ============================================================
# GEOGRAPHIC POINT
# ============================================================

class ZoneGeoPoint(ZoneBaseModel):
    latitude: float = Field(
        ge=-90,
        le=90,
        allow_inf_nan=False,
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        allow_inf_nan=False,
    )


# ============================================================
# GEOFENCE
# ============================================================

class ZoneGeofence(ZoneBaseModel):
    enabled: bool = False

    polygon: list[ZoneGeoPoint] = Field(
        default_factory=list
    )

    coordinate_reference: str = "EPSG:4326"

    @model_validator(mode="after")
    def validate_polygon(self) -> "ZoneGeofence":
        if self.enabled and len(self.polygon) < 3:
            raise ValueError(
                "Enabled geofence requires "
                "at least three vertices"
            )

        if self.polygon:
            unique_points = {
                (point.latitude, point.longitude)
                for point in self.polygon
            }

            if len(unique_points) < 3:
                raise ValueError(
                    "Polygon requires three distinct points"
                )

        return self

    def contains(
        self,
        latitude: float,
        longitude: float,
    ) -> bool:
        """
        Basic point-in-polygon test using ray casting.

        Suitable for small, non-dateline-crossing
        application geofences.

        Use a dedicated geospatial engine for
        production GIS and precise boundaries.
        """

        if not self.enabled or len(self.polygon) < 3:
            return False

        if not (-90 <= latitude <= 90):
            raise ValueError("Invalid latitude")

        if not (-180 <= longitude <= 180):
            raise ValueError("Invalid longitude")

        points = [
            (p.longitude, p.latitude)
            for p in self.polygon
        ]

        inside = False
        j = len(points) - 1

        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]

            # Boundary points are treated as inside.
            cross = (
                (longitude - xi) * (yj - yi)
                - (latitude - yi) * (xj - xi)
            )

            if abs(cross) < 1e-10 and (
                min(xi, xj) <= longitude <= max(xi, xj)
                and min(yi, yj) <= latitude <= max(yi, yj)
            ):
                return True

            intersects = (
                (yi > latitude) != (yj > latitude)
            )

            if intersects:
                x_intersection = (
                    (xj - xi)
                    * (latitude - yi)
                    / (yj - yi)
                    + xi
                )

                if longitude < x_intersection:
                    inside = not inside

            j = i

        return inside


# ============================================================
# ZONE HAZARD
# ============================================================

class ZoneHazard(ZoneBaseModel):
    hazard_id: UUID = Field(
        default_factory=uuid4
    )

    hazard_type: ZoneHazardType

    description: str = Field(min_length=1)

    risk_level: ZoneRiskLevel

    mitigation_description: str | None = None

    active: bool = True

    identified_at: datetime = Field(
        default_factory=utc_now
    )

    reviewed_by: str | None = None

    @field_validator("identified_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_datetime(value)


# ============================================================
# ACCESS POLICY
# ============================================================

class ZoneAccessPolicy(ZoneBaseModel):
    access_level: ZoneAccessLevel = (
        ZoneAccessLevel.EMPLOYEE
    )

    restricted_access: bool = False

    authorized_role_ids: list[str] = Field(
        default_factory=list
    )

    authorized_user_ids: list[str] = Field(
        default_factory=list
    )

    permit_required: bool = False

    permit_type: str | None = None

    escort_required: bool = False

    access_logging_enabled: bool = True

    @model_validator(mode="after")
    def validate_access(self) -> "ZoneAccessPolicy":
        if (
            self.access_level
            == ZoneAccessLevel.RESTRICTED.value
            and not self.restricted_access
        ):
            raise ValueError(
                "Restricted access level requires "
                "restricted_access=True"
            )

        if self.permit_required and not self.permit_type:
            raise ValueError(
                "Permit type is required"
            )

        return self


# ============================================================
# PPE REQUIREMENTS
# ============================================================

class ZonePPERequirements(ZoneBaseModel):
    required_ppe: list[ZonePPEType] = Field(
        default_factory=list
    )

    ppe_detection_enabled: bool = False

    enforcement_policy_reference: str | None = None

    @field_validator("required_ppe")
    @classmethod
    def unique_ppe(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(value))


# ============================================================
# ZONE OCCUPANCY
# ============================================================

class ZoneOccupancy(ZoneBaseModel):
    current_count: int = Field(
        default=0,
        ge=0,
    )

    maximum_capacity: int | None = Field(
        default=None,
        ge=1,
    )

    status: ZoneOccupancyStatus = (
        ZoneOccupancyStatus.UNKNOWN
    )

    source: str | None = None

    measured_at: datetime | None = None

    @field_validator("measured_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @property
    def occupancy_percent(self) -> float | None:
        if self.maximum_capacity is None:
            return None

        return round(
            self.current_count
            / self.maximum_capacity
            * 100,
            2,
        )

    def calculate_status(self) -> str:
        if self.maximum_capacity is None:
            return ZoneOccupancyStatus.UNKNOWN.value

        if self.current_count > self.maximum_capacity:
            return ZoneOccupancyStatus.OVER_CAPACITY.value

        if (
            self.current_count
            >= 0.9 * self.maximum_capacity
        ):
            return ZoneOccupancyStatus.NEAR_CAPACITY.value

        return ZoneOccupancyStatus.NORMAL.value


# ============================================================
# MONITORING INFRASTRUCTURE
# ============================================================

class ZoneMonitoring(ZoneBaseModel):
    monitoring_enabled: bool = True

    status: ZoneMonitoringStatus = (
        ZoneMonitoringStatus.UNKNOWN
    )

    camera_ids: list[str] = Field(
        default_factory=list
    )

    thermal_camera_ids: list[str] = Field(
        default_factory=list
    )

    sensor_ids: list[str] = Field(
        default_factory=list
    )

    gateway_ids: list[str] = Field(
        default_factory=list
    )

    fire_detection_enabled: bool = False
    smoke_detection_enabled: bool = False
    gas_detection_enabled: bool = False

    person_detection_enabled: bool = False
    ppe_detection_enabled: bool = False
    restricted_entry_detection_enabled: bool = False

    last_heartbeat_at: datetime | None = None

    @field_validator("last_heartbeat_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)


# ============================================================
# EMERGENCY INFRASTRUCTURE
# ============================================================

class ZoneEmergencyInfrastructure(ZoneBaseModel):
    emergency_exit_ids: list[str] = Field(
        default_factory=list
    )

    assembly_point_ids: list[str] = Field(
        default_factory=list
    )

    alarm_device_ids: list[str] = Field(
        default_factory=list
    )

    fire_extinguisher_ids: list[str] = Field(
        default_factory=list
    )

    first_aid_station_ids: list[str] = Field(
        default_factory=list
    )

    emergency_plan_reference: str | None = None

    evacuation_route_reference: str | None = None

    emergency_coordinator_ids: list[str] = Field(
        default_factory=list
    )

    last_drill_at: datetime | None = None

    @field_validator("last_drill_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)


# ============================================================
# ZONE HEALTH
# ============================================================

class ZoneHealth(ZoneBaseModel):
    status: ZoneHealthStatus = (
        ZoneHealthStatus.UNKNOWN
    )

    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    active_safety_event_count: int = Field(
        default=0,
        ge=0,
    )

    critical_safety_event_count: int = Field(
        default=0,
        ge=0,
    )

    active_anomaly_count: int = Field(
        default=0,
        ge=0,
    )

    unhealthy_machine_count: int = Field(
        default=0,
        ge=0,
    )

    last_assessed_at: datetime | None = None

    assessment_method: str | None = None

    @field_validator("last_assessed_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_counts(self) -> "ZoneHealth":
        if (
            self.critical_safety_event_count
            > self.active_safety_event_count
        ):
            raise ValueError(
                "Critical event count cannot exceed "
                "active event count"
            )

        return self


# ============================================================
# RISK ASSESSMENT
# ============================================================

class ZoneRiskAssessment(ZoneBaseModel):
    assessment_id: UUID = Field(
        default_factory=uuid4
    )

    risk_level: ZoneRiskLevel

    risk_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    identified_hazards: list[str] = Field(
        default_factory=list
    )

    recommended_controls: list[str] = Field(
        default_factory=list
    )

    assessment_method: str | None = None

    assessed_by: str = Field(min_length=1)

    assessed_at: datetime = Field(
        default_factory=utc_now
    )

    review_status: ZoneReviewStatus = (
        ZoneReviewStatus.PENDING
    )

    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    notes: str | None = None

    @field_validator(
        "assessed_at",
        "reviewed_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_review(
        self,
    ) -> "ZoneRiskAssessment":
        if self.review_status in {
            ZoneReviewStatus.APPROVED.value,
            ZoneReviewStatus.REJECTED.value,
        }:
            if not self.reviewed_by or not self.reviewed_at:
                raise ValueError(
                    "Completed review requires "
                    "reviewer and timestamp"
                )

        return self


# ============================================================
# ZONE EVENT REFERENCE
# ============================================================

class ZoneEventReference(ZoneBaseModel):
    event_id: str = Field(min_length=1)

    event_type: str = Field(min_length=1)

    severity: str | None = None

    detected_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("detected_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_datetime(value)


# ============================================================
# MAIN INDUSTRIAL ZONE
# ============================================================

class IndustrialZone(ZoneBaseModel):
    zone_id: str = Field(min_length=1)

    zone_uuid: UUID = Field(
        default_factory=uuid4
    )

    zone_code: str = Field(
        min_length=1,
        max_length=100,
    )

    zone_name: str = Field(
        min_length=1,
        max_length=250,
    )

    context: ZoneContext

    zone_type: ZoneType

    status: ZoneStatus = ZoneStatus.PLANNED

    risk_level: ZoneRiskLevel = (
        ZoneRiskLevel.UNKNOWN
    )

    description: str | None = None

    geofence: ZoneGeofence = Field(
        default_factory=ZoneGeofence
    )

    access_policy: ZoneAccessPolicy = Field(
        default_factory=ZoneAccessPolicy
    )

    ppe_requirements: ZonePPERequirements = Field(
        default_factory=ZonePPERequirements
    )

    hazards: list[ZoneHazard] = Field(
        default_factory=list
    )

    occupancy: ZoneOccupancy = Field(
        default_factory=ZoneOccupancy
    )

    monitoring: ZoneMonitoring = Field(
        default_factory=ZoneMonitoring
    )

    emergency_infrastructure: (
        ZoneEmergencyInfrastructure
    ) = Field(
        default_factory=ZoneEmergencyInfrastructure
    )

    health: ZoneHealth = Field(
        default_factory=ZoneHealth
    )

    risk_assessments: list[
        ZoneRiskAssessment
    ] = Field(default_factory=list)

    safety_events: list[
        ZoneEventReference
    ] = Field(default_factory=list)

    asset_ids: list[str] = Field(
        default_factory=list
    )

    machine_ids: list[str] = Field(
        default_factory=list
    )

    inspection_ids: list[str] = Field(
        default_factory=list
    )

    maintenance_ids: list[str] = Field(
        default_factory=list
    )

    responsible_user_ids: list[str] = Field(
        default_factory=list
    )

    activated_at: datetime | None = None

    decommissioned_at: datetime | None = None

    created_by: str | None = None

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
    def validate_zone(self) -> "IndustrialZone":
        if self.context.parent_zone_id == self.zone_id:
            raise ValueError(
                "Zone cannot be its own parent"
            )

        if (
            self.zone_type == ZoneType.RESTRICTED.value
            and not self.access_policy.restricted_access
        ):
            raise ValueError(
                "Restricted zones require "
                "restricted access policy"
            )

        collections = (
            ("hazard", [
                item.hazard_id
                for item in self.hazards
            ]),
            ("risk assessment", [
                item.assessment_id
                for item in self.risk_assessments
            ]),
            ("safety event", [
                item.event_id
                for item in self.safety_events
            ]),
            ("asset", self.asset_ids),
            ("machine", self.machine_ids),
        )

        for label, identifiers in collections:
            if len(identifiers) != len(
                set(identifiers)
            ):
                raise ValueError(
                    f"Duplicate {label} identifiers"
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
        if self.status == ZoneStatus.DECOMMISSIONED.value:
            raise ValueError(
                "Decommissioned zone cannot be activated"
            )

        if self.status == ZoneStatus.ACTIVE.value:
            raise ValueError(
                "Zone is already active"
            )

        self.status = ZoneStatus.ACTIVE
        self.activated_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # RESTRICT
    # --------------------------------------------------------

    def restrict(
        self,
        reason: str,
    ) -> None:
        if self.status not in {
            ZoneStatus.ACTIVE.value,
            ZoneStatus.RESTRICTED.value,
        }:
            raise ValueError(
                "Zone cannot be restricted "
                "from its current status"
            )

        if not reason.strip():
            raise ValueError(
                "Restriction reason is required"
            )

        self.status = ZoneStatus.RESTRICTED
        self.metadata["restriction_reason"] = reason
        self._touch()

    # --------------------------------------------------------
    # TEMPORARILY CLOSE
    # --------------------------------------------------------

    def temporarily_close(
        self,
        reason: str,
    ) -> None:
        if self.status == ZoneStatus.DECOMMISSIONED.value:
            raise ValueError(
                "Decommissioned zone cannot be closed"
            )

        if not reason.strip():
            raise ValueError(
                "Closure reason is required"
            )

        self.status = ZoneStatus.TEMPORARILY_CLOSED
        self.metadata["closure_reason"] = reason
        self._touch()

    # --------------------------------------------------------
    # DECOMMISSION
    # --------------------------------------------------------

    def decommission(self) -> None:
        if self.status == ZoneStatus.DECOMMISSIONED.value:
            raise ValueError(
                "Zone already decommissioned"
            )

        self.status = ZoneStatus.DECOMMISSIONED
        self.decommissioned_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # ADD HAZARD
    # --------------------------------------------------------

    def add_hazard(
        self,
        hazard: ZoneHazard,
    ) -> None:
        if any(
            item.hazard_id == hazard.hazard_id
            for item in self.hazards
        ):
            raise ValueError(
                "Hazard already exists"
            )

        self.hazards.append(hazard)
        self._touch()

    # --------------------------------------------------------
    # ADD RISK ASSESSMENT
    # --------------------------------------------------------

    def add_risk_assessment(
        self,
        assessment: ZoneRiskAssessment,
    ) -> None:
        if any(
            item.assessment_id == assessment.assessment_id
            for item in self.risk_assessments
        ):
            raise ValueError(
                "Risk assessment already exists"
            )

        self.risk_assessments.append(assessment)
        self._touch()

    # --------------------------------------------------------
    # ADD SAFETY EVENT
    # --------------------------------------------------------

    def add_safety_event(
        self,
        event: ZoneEventReference,
    ) -> None:
        if any(
            item.event_id == event.event_id
            for item in self.safety_events
        ):
            raise ValueError(
                "Safety event already linked"
            )

        self.safety_events.append(event)
        self._touch()

    # --------------------------------------------------------
    # UPDATE OCCUPANCY
    # --------------------------------------------------------

    def update_occupancy(
        self,
        occupancy: ZoneOccupancy,
    ) -> None:
        if occupancy.measured_at is None:
            raise ValueError(
                "Occupancy measurement time is required"
            )

        occupancy.status = (
            occupancy.calculate_status()
        )

        self.occupancy = occupancy
        self._touch()

    # --------------------------------------------------------
    # UPDATE HEALTH
    # --------------------------------------------------------

    def update_health(
        self,
        health: ZoneHealth,
    ) -> None:
        self.health = health
        self._touch()

    # --------------------------------------------------------
    # REGISTER ASSET
    # --------------------------------------------------------

    def register_asset(
        self,
        asset_id: str,
    ) -> None:
        if not asset_id.strip():
            raise ValueError(
                "Asset ID is required"
            )

        if asset_id in self.asset_ids:
            raise ValueError(
                "Asset already registered"
            )

        self.asset_ids.append(asset_id)
        self._touch()

    # --------------------------------------------------------
    # REGISTER MACHINE
    # --------------------------------------------------------

    def register_machine(
        self,
        machine_id: str,
    ) -> None:
        if not machine_id.strip():
            raise ValueError(
                "Machine ID is required"
            )

        if machine_id in self.machine_ids:
            raise ValueError(
                "Machine already registered"
            )

        self.machine_ids.append(machine_id)
        self._touch()

    # --------------------------------------------------------
    # RISK HELPERS
    # --------------------------------------------------------

    def active_hazard_count(self) -> int:
        return sum(
            hazard.active
            for hazard in self.hazards
        )

    def critical_hazard_count(self) -> int:
        return sum(
            hazard.active
            and hazard.risk_level
            == ZoneRiskLevel.CRITICAL.value
            for hazard in self.hazards
        )

    def requires_safety_review(self) -> bool:
        return (
            self.risk_level in {
                ZoneRiskLevel.HIGH.value,
                ZoneRiskLevel.CRITICAL.value,
            }
            or self.critical_hazard_count() > 0
            or self.health.critical_safety_event_count > 0
            or self.occupancy.status
            == ZoneOccupancyStatus.OVER_CAPACITY.value
        )

    # --------------------------------------------------------
    # GEOFENCE CHECK
    # --------------------------------------------------------

    def contains_location(
        self,
        latitude: float,
        longitude: float,
    ) -> bool:
        return self.geofence.contains(
            latitude,
            longitude,
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class ZoneCreate(ZoneBaseModel):
    zone_id: str = Field(min_length=1)

    zone_code: str = Field(min_length=1)

    zone_name: str = Field(min_length=1)

    context: ZoneContext

    zone_type: ZoneType

    risk_level: ZoneRiskLevel = (
        ZoneRiskLevel.UNKNOWN
    )

    description: str | None = None

    access_policy: ZoneAccessPolicy = Field(
        default_factory=ZoneAccessPolicy
    )

    ppe_requirements: ZonePPERequirements = Field(
        default_factory=ZonePPERequirements
    )


# ============================================================
# UPDATE SCHEMA
# ============================================================

class ZoneUpdate(ZoneBaseModel):
    zone_name: str | None = None

    description: str | None = None

    risk_level: ZoneRiskLevel | None = None

    geofence: ZoneGeofence | None = None

    access_policy: ZoneAccessPolicy | None = None

    ppe_requirements: ZonePPERequirements | None = None

    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class ZoneResponse(IndustrialZone):
    pass


# ============================================================
# SUMMARY
# ============================================================

class ZoneSummary(ZoneBaseModel):
    zone_id: str
    zone_code: str
    zone_name: str

    company_id: str
    site_id: str

    building_id: str | None

    zone_type: ZoneType
    status: ZoneStatus
    risk_level: ZoneRiskLevel

    health_status: ZoneHealthStatus

    restricted_access: bool

    current_occupancy: int
    maximum_occupancy: int | None
    occupancy_percent: float | None

    asset_count: int
    machine_count: int
    sensor_count: int
    camera_count: int

    active_hazard_count: int
    critical_hazard_count: int

    safety_event_count: int

    requires_safety_review: bool

    updated_at: datetime

    @classmethod
    def from_zone(
        cls,
        zone: IndustrialZone,
    ) -> "ZoneSummary":
        return cls(
            zone_id=zone.zone_id,
            zone_code=zone.zone_code,
            zone_name=zone.zone_name,
            company_id=zone.context.company_id,
            site_id=zone.context.site_id,
            building_id=zone.context.building_id,
            zone_type=zone.zone_type,
            status=zone.status,
            risk_level=zone.risk_level,
            health_status=zone.health.status,
            restricted_access=(
                zone.access_policy.restricted_access
            ),
            current_occupancy=(
                zone.occupancy.current_count
            ),
            maximum_occupancy=(
                zone.occupancy.maximum_capacity
            ),
            occupancy_percent=(
                zone.occupancy.occupancy_percent
            ),
            asset_count=len(zone.asset_ids),
            machine_count=len(zone.machine_ids),
            sensor_count=len(
                zone.monitoring.sensor_ids
            ),
            camera_count=(
                len(zone.monitoring.camera_ids)
                + len(zone.monitoring.thermal_camera_ids)
            ),
            active_hazard_count=(
                zone.active_hazard_count()
            ),
            critical_hazard_count=(
                zone.critical_hazard_count()
            ),
            safety_event_count=len(
                zone.safety_events
            ),
            requires_safety_review=(
                zone.requires_safety_review()
            ),
            updated_at=zone.updated_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    zone = IndustrialZone(
        zone_id="ZONE-001",
        zone_code="HIGH-TEMP-01",
        zone_name="High Temperature Motor Area",
        context=ZoneContext(
            company_id="COMPANY-001",
            site_id="SITE-001",
            building_id="BUILDING-001",
            floor_id="GROUND",
            production_line_id="LINE-001",
        ),
        zone_type=ZoneType.HIGH_TEMPERATURE,
        risk_level=ZoneRiskLevel.HIGH,
        description=(
            "Industrial motor testing and "
            "high-temperature equipment area"
        ),
        access_policy=ZoneAccessPolicy(
            access_level=(
                ZoneAccessLevel.AUTHORIZED_PERSONNEL
            ),
            restricted_access=True,
            authorized_role_ids=[
                "maintenance_engineer",
                "safety_officer",
            ],
            permit_required=True,
            permit_type="HOT_WORK_PERMIT",
        ),
        ppe_requirements=ZonePPERequirements(
            required_ppe=[
                ZonePPEType.HELMET,
                ZonePPEType.GLOVES,
                ZonePPEType.SAFETY_SHOES,
                ZonePPEType.SAFETY_GLASSES,
            ],
            ppe_detection_enabled=True,
        ),
        monitoring=ZoneMonitoring(
            monitoring_enabled=True,
            status=ZoneMonitoringStatus.ONLINE,
            camera_ids=["CAM-001"],
            thermal_camera_ids=["THERMAL-001"],
            sensor_ids=[
                "TEMP-101",
                "SMOKE-101",
                "GAS-101",
            ],
            fire_detection_enabled=True,
            smoke_detection_enabled=True,
            gas_detection_enabled=True,
            person_detection_enabled=True,
            ppe_detection_enabled=True,
        ),
    )

    zone.add_hazard(
        ZoneHazard(
            hazard_type=ZoneHazardType.OVERHEATING,
            description=(
                "Potential overheating of motor housing"
            ),
            risk_level=ZoneRiskLevel.HIGH,
            mitigation_description=(
                "Temperature monitoring and "
                "authorized engineering inspection"
            ),
        )
    )

    zone.register_asset("ASSET-101")
    zone.register_machine("MOTOR-101")

    zone.update_occupancy(
        ZoneOccupancy(
            current_count=8,
            maximum_capacity=10,
            measured_at=utc_now(),
            source="person_detection",
        )
    )

    zone.add_safety_event(
        ZoneEventReference(
            event_id="SAFE-2026-001",
            event_type="overheating",
            severity="high",
        )
    )

    zone.activate()

    summary = ZoneSummary.from_zone(zone)

    print(
        json.dumps(
            summary.to_dict(),
            indent=2,
        )
    )

    print(
        "Safety review required:",
        zone.requires_safety_review(),
    )


if __name__ == "__main__":
    example()