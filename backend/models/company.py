"""
app/ai/ml/models/company.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Company / Organization Domain Models

Python 3.11+
Pydantic v2

Dependencies:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant company management
- Company identity and registration
- Industrial facilities and sites
- Company contact information
- Subscription and feature settings
- Safety and maintenance configuration
- Cloud and deployment configuration
- Company lifecycle management
- Data retention configuration
- Tenant isolation identifiers
- Create, update and response schemas
- Pydantic validation
- JSON-compatible serialization

Note:
    These are domain/data-transfer models,
    not SQLAlchemy ORM entities or ML estimators.

    Authentication, authorization, database
    isolation and billing must be enforced
    separately by the application.
"""

from __future__ import annotations

import re
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

class CompanyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    ARCHIVED = "archived"


class IndustryType(str, Enum):
    MANUFACTURING = "manufacturing"
    AUTOMOTIVE = "automotive"
    ENERGY = "energy"
    OIL_GAS = "oil_gas"
    CHEMICAL = "chemical"
    PHARMACEUTICAL = "pharmaceutical"
    FOOD_PROCESSING = "food_processing"
    STEEL = "steel"
    MINING = "mining"
    CONSTRUCTION = "construction"
    LOGISTICS = "logistics"
    UTILITIES = "utilities"
    AEROSPACE = "aerospace"
    ELECTRONICS = "electronics"
    OTHER = "other"


class CompanySize(str, Enum):
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"


class SubscriptionPlan(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class DeploymentType(str, Enum):
    CLOUD = "cloud"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"
    AIR_GAPPED = "air_gapped"


class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    PRIVATE = "private"
    NONE = "none"


class SiteStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNDER_CONSTRUCTION = "under_construction"
    CLOSED = "closed"


class AlertChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


# ============================================================
# BASE MODEL
# ============================================================

class CompanyBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# CONTACT
# ============================================================

class CompanyContact(CompanyBaseModel):
    contact_name: str | None = Field(
        default=None,
        max_length=150,
    )

    email: str | None = Field(
        default=None,
        max_length=254,
    )

    phone: str | None = Field(
        default=None,
        max_length=30,
    )

    department: str | None = Field(
        default=None,
        max_length=100,
    )

    designation: str | None = Field(
        default=None,
        max_length=100,
    )

    @field_validator("email")
    @classmethod
    def validate_email(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip().lower()

        if not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            value,
        ):
            raise ValueError(
                "Invalid email address"
            )

        return value


# ============================================================
# ADDRESS
# ============================================================

class CompanyAddress(CompanyBaseModel):
    address_line_1: str = Field(
        min_length=1,
        max_length=250,
    )

    address_line_2: str | None = None

    city: str = Field(
        min_length=1,
        max_length=100,
    )

    state: str | None = None

    postal_code: str | None = None

    country: str = Field(
        min_length=2,
        max_length=100,
    )

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
# INDUSTRIAL SITE
# ============================================================

class CompanySite(CompanyBaseModel):
    site_id: UUID = Field(
        default_factory=uuid4
    )

    site_code: str = Field(
        min_length=1,
        max_length=100,
    )

    site_name: str = Field(
        min_length=1,
        max_length=200,
    )

    description: str | None = None

    status: SiteStatus = SiteStatus.ACTIVE

    address: CompanyAddress | None = None

    total_assets: int = Field(
        default=0,
        ge=0,
    )

    total_machines: int = Field(
        default=0,
        ge=0,
    )

    total_sensors: int = Field(
        default=0,
        ge=0,
    )

    production_lines: list[str] = Field(
        default_factory=list
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


# ============================================================
# SAFETY CONFIGURATION
# ============================================================

class CompanySafetySettings(CompanyBaseModel):
    enable_fire_detection: bool = True

    enable_smoke_detection: bool = True

    enable_overheating_detection: bool = True

    enable_person_detection: bool = True

    enable_restricted_zone_detection: bool = True

    enable_ppe_detection: bool = True

    enable_emergency_alerts: bool = True

    require_human_review: bool = True

    emergency_contact_email: str | None = None

    emergency_contact_phone: str | None = None

    alert_channels: list[AlertChannel] = Field(
        default_factory=lambda: [
            AlertChannel.DASHBOARD,
            AlertChannel.EMAIL,
        ]
    )

    @field_validator("alert_channels")
    @classmethod
    def unique_alert_channels(
        cls,
        value: list[AlertChannel],
    ) -> list[AlertChannel]:
        return list(dict.fromkeys(value))


# ============================================================
# MAINTENANCE CONFIGURATION
# ============================================================

class CompanyMaintenanceSettings(CompanyBaseModel):
    enable_predictive_maintenance: bool = True

    enable_failure_prediction: bool = True

    enable_rul_prediction: bool = True

    enable_anomaly_detection: bool = True

    enable_visual_inspection: bool = True

    enable_maintenance_forecasting: bool = True

    enable_repair_cost_estimation: bool = True

    enable_replacement_cost_estimation: bool = True

    enable_roi_analysis: bool = True

    maintenance_reminder_days: int = Field(
        default=7,
        ge=0,
        le=365,
    )

    default_inspection_interval_days: int = Field(
        default=30,
        ge=1,
        le=3650,
    )


# ============================================================
# AI CONFIGURATION
# ============================================================

class CompanyAISettings(CompanyBaseModel):
    enable_ml: bool = True
    enable_deep_learning: bool = True
    enable_computer_vision: bool = True

    enable_genai: bool = True
    enable_rag: bool = True
    enable_agents: bool = True

    enable_ai_explanations: bool = True

    require_prediction_review: bool = True

    allow_automatic_work_order_drafts: bool = True

    # Never directly authorize hazardous
    # equipment actions using LLM output.
    allow_automatic_safety_actuation: bool = False

    model_provider: str | None = None
    default_model_name: str | None = None

    @model_validator(mode="after")
    def enforce_safe_actuation(
        self,
    ) -> "CompanyAISettings":
        if self.allow_automatic_safety_actuation:
            raise ValueError(
                "Automatic safety actuation cannot "
                "be enabled through AI settings. "
                "Use separately certified safety "
                "controllers and approved workflows."
            )

        return self


# ============================================================
# SUBSCRIPTION
# ============================================================

class CompanySubscription(CompanyBaseModel):
    plan: SubscriptionPlan = (
        SubscriptionPlan.STARTER
    )

    subscription_start: date | None = None

    subscription_end: date | None = None

    max_users: int = Field(
        default=10,
        ge=1,
    )

    max_assets: int = Field(
        default=100,
        ge=1,
    )

    max_sites: int = Field(
        default=5,
        ge=1,
    )

    max_api_requests_per_day: int = Field(
        default=10000,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_subscription_dates(
        self,
    ) -> "CompanySubscription":
        if (
            self.subscription_start is not None
            and self.subscription_end is not None
            and self.subscription_end
            < self.subscription_start
        ):
            raise ValueError(
                "Subscription end date cannot "
                "precede start date"
            )

        return self


# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================

class CompanyDeploymentSettings(CompanyBaseModel):
    deployment_type: DeploymentType = (
        DeploymentType.CLOUD
    )

    cloud_provider: CloudProvider = (
        CloudProvider.AWS
    )

    region: str | None = None

    enable_encryption_at_rest: bool = True

    enable_encryption_in_transit: bool = True

    enable_audit_logging: bool = True

    enable_model_monitoring: bool = True

    enable_data_backup: bool = True

    backup_retention_days: int = Field(
        default=30,
        ge=1,
        le=3650,
    )

    telemetry_retention_days: int = Field(
        default=365,
        ge=1,
        le=36500,
    )

    @model_validator(mode="after")
    def validate_deployment(
        self,
    ) -> "CompanyDeploymentSettings":
        if (
            self.deployment_type
            == DeploymentType.AIR_GAPPED.value
            and self.cloud_provider
            not in {
                CloudProvider.NONE.value,
                CloudProvider.PRIVATE.value,
            }
        ):
            raise ValueError(
                "Air-gapped deployments require "
                "private or no cloud provider"
            )

        return self


# ============================================================
# COMPANY STATISTICS
# ============================================================

class CompanyStatistics(CompanyBaseModel):
    total_sites: int = Field(
        default=0,
        ge=0,
    )

    total_assets: int = Field(
        default=0,
        ge=0,
    )

    total_machines: int = Field(
        default=0,
        ge=0,
    )

    total_sensors: int = Field(
        default=0,
        ge=0,
    )

    total_users: int = Field(
        default=0,
        ge=0,
    )

    active_alerts: int = Field(
        default=0,
        ge=0,
    )

    open_incidents: int = Field(
        default=0,
        ge=0,
    )

    open_work_orders: int = Field(
        default=0,
        ge=0,
    )

    last_updated_at: datetime | None = None


# ============================================================
# MAIN COMPANY MODEL
# ============================================================

class Company(CompanyBaseModel):
    company_id: UUID = Field(
        default_factory=uuid4
    )

    company_code: str = Field(
        min_length=2,
        max_length=50,
    )

    company_name: str = Field(
        min_length=2,
        max_length=200,
    )

    legal_name: str | None = None

    industry: IndustryType

    company_size: CompanySize = (
        CompanySize.SMALL
    )

    status: CompanyStatus = (
        CompanyStatus.TRIAL
    )

    registration_number: str | None = None

    tax_identifier: str | None = None

    website: str | None = None

    description: str | None = None

    headquarters: CompanyAddress | None = None

    primary_contact: CompanyContact | None = None

    sites: list[CompanySite] = Field(
        default_factory=list
    )

    subscription: CompanySubscription = Field(
        default_factory=CompanySubscription
    )

    safety_settings: CompanySafetySettings = Field(
        default_factory=CompanySafetySettings
    )

    maintenance_settings: (
        CompanyMaintenanceSettings
    ) = Field(
        default_factory=CompanyMaintenanceSettings
    )

    ai_settings: CompanyAISettings = Field(
        default_factory=CompanyAISettings
    )

    deployment_settings: (
        CompanyDeploymentSettings
    ) = Field(
        default_factory=CompanyDeploymentSettings
    )

    statistics: CompanyStatistics = Field(
        default_factory=CompanyStatistics
    )

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

    @field_validator("company_code")
    @classmethod
    def validate_company_code(
        cls,
        value: str,
    ) -> str:
        value = value.upper()

        if not re.fullmatch(
            r"[A-Z0-9][A-Z0-9_-]*",
            value,
        ):
            raise ValueError(
                "company_code may contain only "
                "letters, numbers, underscores "
                "and hyphens"
            )

        return value

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
    def validate_unique_sites(
        self,
    ) -> "Company":
        codes = [
            site.site_code
            for site in self.sites
        ]

        if len(codes) != len(set(codes)):
            raise ValueError(
                "Duplicate site codes"
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )

    def activate(self) -> None:
        self.status = CompanyStatus.ACTIVE
        self.updated_at = datetime.now(
            timezone.utc
        )

    def suspend(self) -> None:
        self.status = CompanyStatus.SUSPENDED
        self.updated_at = datetime.now(
            timezone.utc
        )

    def archive(self) -> None:
        self.status = CompanyStatus.ARCHIVED
        self.updated_at = datetime.now(
            timezone.utc
        )

    def add_site(
        self,
        site: CompanySite,
    ) -> None:
        if any(
            existing.site_code == site.site_code
            for existing in self.sites
        ):
            raise ValueError(
                "Site code already exists"
            )

        if (
            len(self.sites)
            >= self.subscription.max_sites
        ):
            raise ValueError(
                "Company site limit reached"
            )

        self.sites.append(site)

        self.statistics.total_sites = (
            len(self.sites)
        )

        self.updated_at = datetime.now(
            timezone.utc
        )

    def remove_site(
        self,
        site_id: UUID,
    ) -> bool:
        original_count = len(self.sites)

        self.sites = [
            site
            for site in self.sites
            if site.site_id != site_id
        ]

        removed = (
            len(self.sites) < original_count
        )

        if removed:
            self.statistics.total_sites = (
                len(self.sites)
            )

            self.updated_at = datetime.now(
                timezone.utc
            )

        return removed

    def get_site(
        self,
        site_id: UUID,
    ) -> CompanySite | None:
        return next(
            (
                site
                for site in self.sites
                if site.site_id == site_id
            ),
            None,
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class CompanyCreate(CompanyBaseModel):
    company_code: str = Field(
        min_length=2,
        max_length=50,
    )

    company_name: str = Field(
        min_length=2,
        max_length=200,
    )

    industry: IndustryType

    company_size: CompanySize = (
        CompanySize.SMALL
    )

    legal_name: str | None = None

    registration_number: str | None = None

    website: str | None = None

    headquarters: CompanyAddress | None = None

    primary_contact: CompanyContact | None = None


# ============================================================
# UPDATE SCHEMA
# ============================================================

class CompanyUpdate(CompanyBaseModel):
    company_name: str | None = None

    legal_name: str | None = None

    industry: IndustryType | None = None

    company_size: CompanySize | None = None

    website: str | None = None

    description: str | None = None

    headquarters: CompanyAddress | None = None

    primary_contact: CompanyContact | None = None

    tags: list[str] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class CompanyResponse(Company):
    pass


# ============================================================
# COMPANY SUMMARY
# ============================================================

class CompanySummary(CompanyBaseModel):
    company_id: UUID

    company_code: str
    company_name: str

    industry: IndustryType
    status: CompanyStatus

    subscription_plan: SubscriptionPlan

    total_sites: int
    total_assets: int
    total_users: int

    created_at: datetime

    @classmethod
    def from_company(
        cls,
        company: Company,
    ) -> "CompanySummary":
        return cls(
            company_id=company.company_id,
            company_code=company.company_code,
            company_name=company.company_name,
            industry=company.industry,
            status=company.status,
            subscription_plan=(
                company.subscription.plan
            ),
            total_sites=len(company.sites),
            total_assets=(
                company.statistics.total_assets
            ),
            total_users=(
                company.statistics.total_users
            ),
            created_at=company.created_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    company = Company(
        company_code="IND-AI-001",
        company_name="Example Industrial AI Ltd",
        industry=IndustryType.MANUFACTURING,
        company_size=CompanySize.MEDIUM,
        status=CompanyStatus.ACTIVE,
        headquarters=CompanyAddress(
            address_line_1="Industrial Park",
            city="Kochi",
            state="Kerala",
            postal_code="682030",
            country="India",
        ),
        primary_contact=CompanyContact(
            contact_name="Operations Manager",
            email="operations@example.com",
            department="Operations",
        ),
        sites=[
            CompanySite(
                site_code="SITE-001",
                site_name="Manufacturing Plant A",
                status=SiteStatus.ACTIVE,
                production_lines=[
                    "LINE-01",
                    "LINE-02",
                ],
            )
        ],
        subscription=CompanySubscription(
            plan=SubscriptionPlan.ENTERPRISE,
            max_users=500,
            max_assets=10000,
            max_sites=50,
        ),
        deployment_settings=(
            CompanyDeploymentSettings(
                deployment_type=DeploymentType.CLOUD,
                cloud_provider=CloudProvider.AWS,
                region="ap-south-1",
            )
        ),
    )

    company.activate()

    company.add_site(
        CompanySite(
            site_code="SITE-002",
            site_name="Manufacturing Plant B",
        )
    )

    print(
        json.dumps(
            company.to_dict(),
            indent=2,
        )
    )

    summary = CompanySummary.from_company(
        company
    )

    print(
        json.dumps(
            summary.model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()