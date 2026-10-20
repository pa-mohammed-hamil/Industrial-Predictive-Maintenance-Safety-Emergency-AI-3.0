"""
app/ai/ml/models/user.py

Industrial AI User & Access Management Models

Python 3.11+
Pydantic v2

Features:
- Multi-tenant user management
- Role-based access control (RBAC)
- User profiles and organization membership
- Site-level access permissions
- Maintenance engineers and technicians
- Safety officers and emergency responders
- AI agents and service accounts
- MFA and security settings
- Account lifecycle management
- Login and session metadata
- User notification preferences
- Create, update, response and summary schemas

Security:
- Never store plaintext passwords
- Never expose password hashes in API responses
- Enforce tenant isolation in the service layer
- Enforce permissions at API and database boundaries
- Verify identity using a trusted authentication service
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
    SecretStr,
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

class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    COMPANY_ADMIN = "company_admin"
    SITE_ADMIN = "site_admin"

    PLANT_MANAGER = "plant_manager"
    OPERATIONS_MANAGER = "operations_manager"

    MAINTENANCE_MANAGER = "maintenance_manager"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    TECHNICIAN = "technician"

    SAFETY_MANAGER = "safety_manager"
    SAFETY_OFFICER = "safety_officer"
    EMERGENCY_RESPONDER = "emergency_responder"

    INSPECTION_ENGINEER = "inspection_engineer"
    RELIABILITY_ENGINEER = "reliability_engineer"

    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    AI_ENGINEER = "ai_engineer"

    FINANCE_MANAGER = "finance_manager"
    ANALYST = "analyst"
    AUDITOR = "auditor"

    VIEWER = "viewer"

    SERVICE_ACCOUNT = "service_account"
    AI_AGENT = "ai_agent"


class UserStatus(str, Enum):
    INVITED = "invited"
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    DISABLED = "disabled"
    DELETED = "deleted"


class UserType(str, Enum):
    HUMAN = "human"
    SERVICE = "service"
    AI_AGENT = "ai_agent"


class Permission(str, Enum):
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"

    COMPANY_READ = "company:read"
    COMPANY_MANAGE = "company:manage"

    SITE_READ = "site:read"
    SITE_MANAGE = "site:manage"

    ASSET_READ = "asset:read"
    ASSET_MANAGE = "asset:manage"

    MACHINE_READ = "machine:read"
    MACHINE_MANAGE = "machine:manage"

    SENSOR_READ = "sensor:read"
    SENSOR_MANAGE = "sensor:manage"

    TELEMETRY_READ = "telemetry:read"
    TELEMETRY_INGEST = "telemetry:ingest"

    PREDICTION_READ = "prediction:read"
    PREDICTION_RUN = "prediction:run"

    MAINTENANCE_READ = "maintenance:read"
    MAINTENANCE_CREATE = "maintenance:create"
    MAINTENANCE_UPDATE = "maintenance:update"
    MAINTENANCE_APPROVE = "maintenance:approve"

    INSPECTION_READ = "inspection:read"
    INSPECTION_CREATE = "inspection:create"
    INSPECTION_APPROVE = "inspection:approve"

    SAFETY_READ = "safety:read"
    SAFETY_CREATE = "safety:create"
    SAFETY_REVIEW = "safety:review"

    EMERGENCY_READ = "emergency:read"
    EMERGENCY_MANAGE = "emergency:manage"

    INCIDENT_READ = "incident:read"
    INCIDENT_MANAGE = "incident:manage"

    COST_READ = "cost:read"
    COST_MANAGE = "cost:manage"

    REPORT_READ = "report:read"
    REPORT_CREATE = "report:create"

    NOTIFICATION_READ = "notification:read"
    NOTIFICATION_MANAGE = "notification:manage"

    MODEL_READ = "model:read"
    MODEL_DEPLOY = "model:deploy"

    AUDIT_READ = "audit:read"

    SETTINGS_MANAGE = "settings:manage"


class AuthenticationProvider(str, Enum):
    LOCAL = "local"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    AZURE_AD = "azure_ad"
    OKTA = "okta"
    LDAP = "ldap"
    SAML = "saml"
    OIDC = "oidc"
    SERVICE_TOKEN = "service_token"


class MFAType(str, Enum):
    NONE = "none"
    TOTP = "totp"
    SECURITY_KEY = "security_key"
    PASSKEY = "passkey"


class NotificationPreferenceChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    DASHBOARD = "dashboard"


# ============================================================
# BASE MODEL
# ============================================================

class UserBaseModel(BaseModel):
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
# USER PROFILE
# ============================================================

class UserProfile(UserBaseModel):
    first_name: str = Field(
        min_length=1,
        max_length=100,
    )

    last_name: str = Field(
        min_length=1,
        max_length=100,
    )

    display_name: str | None = None

    job_title: str | None = None

    department: str | None = None

    employee_id: str | None = None

    phone_number: str | None = None

    avatar_reference: str | None = None

    timezone_name: str = "UTC"

    language: str = "en"

    @property
    def full_name(self) -> str:
        return (
            f"{self.first_name} "
            f"{self.last_name}"
        ).strip()


# ============================================================
# ORGANIZATION MEMBERSHIP
# ============================================================

class UserOrganizationMembership(UserBaseModel):
    company_id: str = Field(min_length=1)

    roles: list[UserRole] = Field(
        default_factory=lambda: [UserRole.VIEWER]
    )

    site_ids: list[str] = Field(
        default_factory=list
    )

    all_sites_access: bool = False

    active: bool = True

    joined_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("roles")
    @classmethod
    def unique_roles(
        cls,
        value: list[str],
    ) -> list[str]:
        if not value:
            raise ValueError(
                "At least one role is required"
            )

        return list(dict.fromkeys(value))

    @field_validator("site_ids")
    @classmethod
    def unique_sites(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(value))


# ============================================================
# PERMISSION GRANT
# ============================================================

class UserPermissionGrant(UserBaseModel):
    permission: Permission

    company_id: str = Field(min_length=1)

    site_id: str | None = None

    granted_by: str | None = None

    granted_at: datetime = Field(
        default_factory=utc_now
    )

    expires_at: datetime | None = None

    @field_validator(
        "granted_at",
        "expires_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_expiration(
        self,
    ) -> "UserPermissionGrant":
        if (
            self.expires_at is not None
            and self.expires_at <= self.granted_at
        ):
            raise ValueError(
                "Permission expiration must follow grant"
            )

        return self

    def is_active(
        self,
        reference_time: datetime | None = None,
    ) -> bool:
        now = reference_time or utc_now()
        validate_datetime(now)

        return (
            self.expires_at is None
            or now < self.expires_at
        )


# ============================================================
# AUTHENTICATION
# ============================================================

class UserAuthentication(UserBaseModel):
    provider: AuthenticationProvider = (
        AuthenticationProvider.LOCAL
    )

    provider_subject_id: str | None = None

    password_hash: SecretStr | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )

    email_verified: bool = False

    email_verified_at: datetime | None = None

    mfa_enabled: bool = False

    mfa_type: MFAType = MFAType.NONE

    last_password_changed_at: datetime | None = None

    last_login_at: datetime | None = None

    failed_login_attempts: int = Field(
        default=0,
        ge=0,
    )

    locked_until: datetime | None = None

    @field_validator(
        "email_verified_at",
        "last_password_changed_at",
        "last_login_at",
        "locked_until",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_authentication(
        self,
    ) -> "UserAuthentication":
        if self.mfa_enabled and (
            self.mfa_type == MFAType.NONE.value
        ):
            raise ValueError(
                "MFA type required when MFA is enabled"
            )

        if (
            not self.mfa_enabled
            and self.mfa_type != MFAType.NONE.value
        ):
            raise ValueError(
                "MFA type must be none when disabled"
            )

        if (
            self.email_verified_at is not None
            and not self.email_verified
        ):
            raise ValueError(
                "Verification timestamp requires "
                "email_verified=True"
            )

        return self


# ============================================================
# NOTIFICATION PREFERENCES
# ============================================================

class UserNotificationPreferences(UserBaseModel):
    enabled: bool = True

    channels: list[
        NotificationPreferenceChannel
    ] = Field(
        default_factory=lambda: [
            NotificationPreferenceChannel.DASHBOARD,
            NotificationPreferenceChannel.EMAIL,
        ]
    )

    maintenance_alerts: bool = True
    failure_alerts: bool = True
    anomaly_alerts: bool = True

    safety_alerts: bool = True
    emergency_alerts: bool = True

    inspection_reminders: bool = True
    report_notifications: bool = True

    daily_digest: bool = False

    @field_validator("channels")
    @classmethod
    def unique_channels(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(value))


# ============================================================
# USER SESSION METADATA
# ============================================================

class UserSession(UserBaseModel):
    session_id: UUID = Field(
        default_factory=uuid4
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    expires_at: datetime

    last_activity_at: datetime | None = None

    ip_address: str | None = None

    user_agent: str | None = None

    revoked: bool = False

    revoked_at: datetime | None = None

    @field_validator(
        "created_at",
        "expires_at",
        "last_activity_at",
        "revoked_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_session(
        self,
    ) -> "UserSession":
        if self.expires_at <= self.created_at:
            raise ValueError(
                "Session expiration must follow creation"
            )

        if self.revoked and self.revoked_at is None:
            raise ValueError(
                "Revoked session requires revoked_at"
            )

        return self

    def is_active(
        self,
        reference_time: datetime | None = None,
    ) -> bool:
        now = reference_time or utc_now()
        validate_datetime(now)

        return (
            not self.revoked
            and now < self.expires_at
        )


# ============================================================
# MAIN INDUSTRIAL USER
# ============================================================

class IndustrialUser(UserBaseModel):
    user_id: UUID = Field(
        default_factory=uuid4
    )

    username: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-zA-Z0-9._-]+$",
    )

    email: str = Field(
        min_length=3,
        max_length=254,
    )

    user_type: UserType = UserType.HUMAN

    status: UserStatus = UserStatus.INVITED

    profile: UserProfile

    memberships: list[
        UserOrganizationMembership
    ] = Field(default_factory=list)

    permission_grants: list[
        UserPermissionGrant
    ] = Field(default_factory=list)

    authentication: UserAuthentication = Field(
        default_factory=UserAuthentication,
        exclude=True,
        repr=False,
    )

    notification_preferences: (
        UserNotificationPreferences
    ) = Field(
        default_factory=UserNotificationPreferences
    )

    sessions: list[UserSession] = Field(
        default_factory=list,
        exclude=True,
        repr=False,
    )

    created_by: str | None = None

    activated_at: datetime | None = None
    suspended_at: datetime | None = None
    deleted_at: datetime | None = None

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

    @field_validator("email")
    @classmethod
    def validate_email(
        cls,
        value: str,
    ) -> str:
        value = value.lower()

        if (
            value.count("@") != 1
            or "." not in value.split("@")[1]
            or any(char.isspace() for char in value)
        ):
            raise ValueError(
                "Invalid email address"
            )

        return value

    @field_validator(
        "activated_at",
        "suspended_at",
        "deleted_at",
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
    def validate_user(
        self,
    ) -> "IndustrialUser":
        company_ids = [
            item.company_id
            for item in self.memberships
        ]

        if len(company_ids) != len(
            set(company_ids)
        ):
            raise ValueError(
                "Duplicate company memberships"
            )

        for grant in self.permission_grants:
            if grant.company_id not in company_ids:
                raise ValueError(
                    "Permission grant references "
                    "company without membership"
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
        if self.status == UserStatus.DELETED.value:
            raise ValueError(
                "Deleted user cannot be activated"
            )

        if self.status == UserStatus.ACTIVE.value:
            raise ValueError(
                "User is already active"
            )

        if not self.authentication.email_verified:
            raise ValueError(
                "Email verification required"
            )

        self.status = UserStatus.ACTIVE
        self.activated_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # SUSPEND
    # --------------------------------------------------------

    def suspend(self) -> None:
        if self.status != UserStatus.ACTIVE.value:
            raise ValueError(
                "Only active users can be suspended"
            )

        self.status = UserStatus.SUSPENDED
        self.suspended_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # DISABLE
    # --------------------------------------------------------

    def disable(self) -> None:
        if self.status == UserStatus.DELETED.value:
            raise ValueError(
                "Deleted user cannot be disabled"
            )

        self.status = UserStatus.DISABLED
        self._touch()

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    def soft_delete(self) -> None:
        self.status = UserStatus.DELETED
        self.deleted_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # ORGANIZATION MEMBERSHIP
    # --------------------------------------------------------

    def add_membership(
        self,
        membership: UserOrganizationMembership,
    ) -> None:
        if any(
            item.company_id == membership.company_id
            for item in self.memberships
        ):
            raise ValueError(
                "Company membership already exists"
            )

        self.memberships.append(membership)
        self._touch()

    def get_membership(
        self,
        company_id: str,
    ) -> UserOrganizationMembership | None:
        return next(
            (
                membership
                for membership in self.memberships
                if membership.company_id == company_id
            ),
            None,
        )

    # --------------------------------------------------------
    # ROLE MANAGEMENT
    # --------------------------------------------------------

    def assign_role(
        self,
        company_id: str,
        role: UserRole,
    ) -> None:
        membership = self.get_membership(company_id)

        if membership is None:
            raise ValueError(
                "Company membership not found"
            )

        role_value = (
            role.value
            if isinstance(role, UserRole)
            else role
        )

        if role_value not in membership.roles:
            membership.roles.append(role_value)
            self._touch()

    # --------------------------------------------------------
    # SITE ACCESS
    # --------------------------------------------------------

    def has_site_access(
        self,
        company_id: str,
        site_id: str,
    ) -> bool:
        if self.status != UserStatus.ACTIVE.value:
            return False

        membership = self.get_membership(company_id)

        if membership is None or not membership.active:
            return False

        return (
            membership.all_sites_access
            or site_id in membership.site_ids
        )

    # --------------------------------------------------------
    # PERMISSION MANAGEMENT
    # --------------------------------------------------------

    def grant_permission(
        self,
        grant: UserPermissionGrant,
    ) -> None:
        membership = self.get_membership(
            grant.company_id
        )

        if membership is None or not membership.active:
            raise ValueError(
                "Active company membership required"
            )

        if (
            grant.site_id is not None
            and not (
                membership.all_sites_access
                or grant.site_id in membership.site_ids
            )
        ):
            raise ValueError(
                "Grant references unauthorized site"
            )

        self.permission_grants.append(grant)
        self._touch()

    def has_permission(
        self,
        permission: Permission,
        company_id: str,
        site_id: str | None = None,
    ) -> bool:
        """
        Checks explicit permission grants only.

        Role-derived permissions and administrative
        overrides must be resolved by the authorization
        service using an audited RBAC policy.
        """

        if self.status != UserStatus.ACTIVE.value:
            return False

        membership = self.get_membership(company_id)

        if membership is None or not membership.active:
            return False

        if site_id is not None and not (
            membership.all_sites_access
            or site_id in membership.site_ids
        ):
            return False

        permission_value = (
            permission.value
            if isinstance(permission, Permission)
            else permission
        )

        return any(
            grant.company_id == company_id
            and grant.permission == permission_value
            and (
                grant.site_id is None
                or (
                    site_id is not None
                    and grant.site_id == site_id
                )
            )
            and grant.is_active()
            for grant in self.permission_grants
        )

    # --------------------------------------------------------
    # ROLE HELPERS
    # --------------------------------------------------------

    def has_role(
        self,
        company_id: str,
        role: UserRole,
    ) -> bool:
        membership = self.get_membership(company_id)

        if membership is None or not membership.active:
            return False

        role_value = (
            role.value
            if isinstance(role, UserRole)
            else role
        )

        return role_value in membership.roles

    # --------------------------------------------------------
    # SAFETY RESPONSIBILITY
    # --------------------------------------------------------

    def is_safety_officer(
        self,
        company_id: str,
    ) -> bool:
        return (
            self.has_role(
                company_id,
                UserRole.SAFETY_OFFICER,
            )
            or self.has_role(
                company_id,
                UserRole.SAFETY_MANAGER,
            )
        )

    # --------------------------------------------------------
    # ACCOUNT HELPERS
    # --------------------------------------------------------

    def is_active(self) -> bool:
        return self.status == UserStatus.ACTIVE.value

    def company_count(self) -> int:
        return len(self.memberships)

    def active_session_count(self) -> int:
        return sum(
            session.is_active()
            for session in self.sessions
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class UserCreate(UserBaseModel):
    username: str = Field(
        min_length=3,
        max_length=100,
    )

    email: str = Field(min_length=3)

    user_type: UserType = UserType.HUMAN

    profile: UserProfile

    memberships: list[
        UserOrganizationMembership
    ] = Field(default_factory=list)

    created_by: str | None = None


# ============================================================
# UPDATE SCHEMA
# ============================================================

class UserUpdate(UserBaseModel):
    profile: UserProfile | None = None

    notification_preferences: (
        UserNotificationPreferences | None
    ) = None

    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# PUBLIC RESPONSE SCHEMA
# ============================================================

class UserResponse(UserBaseModel):
    user_id: UUID

    username: str
    email: str

    user_type: UserType
    status: UserStatus

    profile: UserProfile

    memberships: list[
        UserOrganizationMembership
    ]

    notification_preferences: (
        UserNotificationPreferences
    )

    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_user(
        cls,
        user: IndustrialUser,
    ) -> "UserResponse":
        return cls(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            user_type=user.user_type,
            status=user.status,
            profile=user.profile,
            memberships=user.memberships,
            notification_preferences=(
                user.notification_preferences
            ),
            created_at=user.created_at,
            updated_at=user.updated_at,
        )


# ============================================================
# USER SUMMARY
# ============================================================

class UserSummary(UserBaseModel):
    user_id: UUID

    username: str
    email: str

    full_name: str

    user_type: UserType
    status: UserStatus

    company_count: int
    roles: list[str]

    mfa_enabled: bool

    created_at: datetime

    @classmethod
    def from_user(
        cls,
        user: IndustrialUser,
    ) -> "UserSummary":
        roles = sorted({
            role
            for membership in user.memberships
            for role in membership.roles
        })

        return cls(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            full_name=user.profile.full_name,
            user_type=user.user_type,
            status=user.status,
            company_count=user.company_count(),
            roles=roles,
            mfa_enabled=(
                user.authentication.mfa_enabled
            ),
            created_at=user.created_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    user = IndustrialUser(
        username="maintenance.engineer",
        email="engineer@example.com",
        profile=UserProfile(
            first_name="Maintenance",
            last_name="Engineer",
            job_title="Predictive Maintenance Engineer",
            department="Maintenance",
            timezone_name="Asia/Kolkata",
        ),
        memberships=[
            UserOrganizationMembership(
                company_id="COMPANY-001",
                roles=[
                    UserRole.MAINTENANCE_ENGINEER,
                    UserRole.TECHNICIAN,
                ],
                site_ids=[
                    "SITE-001",
                    "SITE-002",
                ],
            )
        ],
        authentication=UserAuthentication(
            provider=AuthenticationProvider.LOCAL,
            email_verified=True,
            email_verified_at=utc_now(),
            mfa_enabled=True,
            mfa_type=MFAType.TOTP,
        ),
    )

    user.activate()

    user.grant_permission(
        UserPermissionGrant(
            permission=Permission.MAINTENANCE_READ,
            company_id="COMPANY-001",
            site_id="SITE-001",
            granted_by="ADMIN-001",
        )
    )

    print(
        "Has site access:",
        user.has_site_access(
            "COMPANY-001",
            "SITE-001",
        ),
    )

    print(
        "Can read maintenance:",
        user.has_permission(
            Permission.MAINTENANCE_READ,
            "COMPANY-001",
            "SITE-001",
        ),
    )

    print(
        json.dumps(
            UserResponse.from_user(
                user
            ).to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()