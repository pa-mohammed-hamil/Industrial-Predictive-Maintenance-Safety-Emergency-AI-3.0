# app/config/security.py

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4

import jwt
from fastapi import (
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from jwt import (
    ExpiredSignatureError,
    InvalidTokenError,
)
from pwdlib import PasswordHash

logger = logging.getLogger(__name__)


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True, frozen=True)
class SecurityConfig:
    secret_key: str = os.getenv(
        "JWT_SECRET_KEY",
        "",
    )

    algorithm: str = os.getenv(
        "JWT_ALGORITHM",
        "HS256",
    )

    issuer: str = os.getenv(
        "JWT_ISSUER",
        "industrial-ai",
    )

    audience: str = os.getenv(
        "JWT_AUDIENCE",
        "industrial-ai-api",
    )

    access_token_minutes: int = int(
        os.getenv(
            "ACCESS_TOKEN_EXPIRE_MINUTES",
            "30",
        )
    )

    refresh_token_days: int = int(
        os.getenv(
            "REFRESH_TOKEN_EXPIRE_DAYS",
            "7",
        )
    )

    api_key_pepper: str = os.getenv(
        "API_KEY_PEPPER",
        "",
    )

    def validate(self) -> None:
        if len(self.secret_key) < 32:
            raise RuntimeError(
                "JWT_SECRET_KEY must contain at least "
                "32 characters."
            )

        if len(self.api_key_pepper) < 16:
            raise RuntimeError(
                "API_KEY_PEPPER must contain at least "
                "16 characters."
            )


security_config = SecurityConfig()


# =========================================================
# Roles / permissions
# =========================================================

class Role(str, Enum):
    ADMIN = "admin"
    SAFETY_MANAGER = "safety_manager"
    MAINTENANCE_MANAGER = "maintenance_manager"
    ENGINEER = "engineer"
    TECHNICIAN = "technician"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(str, Enum):
    ASSET_READ = "asset:read"
    ASSET_WRITE = "asset:write"

    TELEMETRY_READ = "telemetry:read"

    SENSOR_READ = "sensor:read"
    SENSOR_WRITE = "sensor:write"

    PREDICTION_READ = "prediction:read"
    PREDICTION_RUN = "prediction:run"

    MAINTENANCE_READ = "maintenance:read"
    MAINTENANCE_WRITE = "maintenance:write"

    WORK_ORDER_READ = "work_order:read"
    WORK_ORDER_WRITE = "work_order:write"

    SAFETY_READ = "safety:read"
    SAFETY_WRITE = "safety:write"

    INCIDENT_READ = "incident:read"
    INCIDENT_WRITE = "incident:write"

    EMERGENCY_READ = "emergency:read"
    EMERGENCY_CONTROL = "emergency:control"

    COST_READ = "cost:read"

    REPORT_READ = "report:read"
    REPORT_CREATE = "report:create"

    USER_READ = "user:read"
    USER_WRITE = "user:write"

    ADMIN = "admin:*"


ROLE_PERMISSIONS: dict[
    Role,
    set[Permission],
] = {
    Role.ADMIN: set(Permission),

    Role.SAFETY_MANAGER: {
        Permission.ASSET_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.PREDICTION_READ,
        Permission.SAFETY_READ,
        Permission.SAFETY_WRITE,
        Permission.INCIDENT_READ,
        Permission.INCIDENT_WRITE,
        Permission.EMERGENCY_READ,
        Permission.EMERGENCY_CONTROL,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
    },

    Role.MAINTENANCE_MANAGER: {
        Permission.ASSET_READ,
        Permission.ASSET_WRITE,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.PREDICTION_READ,
        Permission.PREDICTION_RUN,
        Permission.MAINTENANCE_READ,
        Permission.MAINTENANCE_WRITE,
        Permission.WORK_ORDER_READ,
        Permission.WORK_ORDER_WRITE,
        Permission.COST_READ,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
    },

    Role.ENGINEER: {
        Permission.ASSET_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.SENSOR_WRITE,
        Permission.PREDICTION_READ,
        Permission.PREDICTION_RUN,
        Permission.MAINTENANCE_READ,
        Permission.WORK_ORDER_READ,
        Permission.SAFETY_READ,
        Permission.INCIDENT_READ,
        Permission.COST_READ,
        Permission.REPORT_READ,
    },

    Role.TECHNICIAN: {
        Permission.ASSET_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.PREDICTION_READ,
        Permission.MAINTENANCE_READ,
        Permission.MAINTENANCE_WRITE,
        Permission.WORK_ORDER_READ,
        Permission.WORK_ORDER_WRITE,
        Permission.SAFETY_READ,
    },

    Role.OPERATOR: {
        Permission.ASSET_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.SAFETY_READ,
        Permission.INCIDENT_READ,
        Permission.INCIDENT_WRITE,
        Permission.EMERGENCY_READ,
    },

    Role.ANALYST: {
        Permission.ASSET_READ,
        Permission.TELEMETRY_READ,
        Permission.PREDICTION_READ,
        Permission.MAINTENANCE_READ,
        Permission.COST_READ,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
    },

    Role.VIEWER: {
        Permission.ASSET_READ,
        Permission.TELEMETRY_READ,
        Permission.SAFETY_READ,
        Permission.REPORT_READ,
    },
}


# =========================================================
# Password hashing
# =========================================================

password_hash = PasswordHash.recommended()


def hash_password(
    password: str,
) -> str:
    if len(password) < 8:
        raise ValueError(
            "Password must contain at least "
            "8 characters."
        )

    return password_hash.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    try:
        return password_hash.verify(
            plain_password,
            hashed_password,
        )
    except Exception:
        logger.warning(
            "Password verification failed"
        )
        return False


def password_needs_rehash(
    hashed_password: str,
) -> bool:
    return password_hash.check_needs_rehash(
        hashed_password
    )


# =========================================================
# Token model
# =========================================================

class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass(slots=True, frozen=True)
class TokenPayload:
    subject: str
    token_type: TokenType
    token_id: str

    roles: tuple[str, ...]
    permissions: tuple[str, ...]

    issued_at: datetime
    expires_at: datetime

    issuer: str
    audience: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.subject,
            "type": self.token_type.value,
            "jti": self.token_id,
            "roles": list(self.roles),
            "permissions": list(
                self.permissions
            ),
            "iat": int(
                self.issued_at.timestamp()
            ),
            "exp": int(
                self.expires_at.timestamp()
            ),
            "iss": self.issuer,
            "aud": self.audience,
        }


# =========================================================
# Permission calculation
# =========================================================

def permissions_for_roles(
    roles: Iterable[str | Role],
) -> set[Permission]:

    result: set[Permission] = set()

    for role_value in roles:
        try:
            role = (
                role_value
                if isinstance(role_value, Role)
                else Role(role_value)
            )
        except ValueError:
            continue

        result.update(
            ROLE_PERMISSIONS.get(
                role,
                set(),
            )
        )

    return result


# =========================================================
# JWT creation
# =========================================================

def _create_token(
    *,
    subject: str,
    token_type: TokenType,
    roles: Iterable[str | Role],
    expires_delta: timedelta,
) -> tuple[str, TokenPayload]:

    security_config.validate()

    now = datetime.now(timezone.utc)

    expires_at = (
        now + expires_delta
    )

    normalized_roles = tuple(
        role.value
        if isinstance(role, Role)
        else str(role)
        for role in roles
    )

    permissions = tuple(
        sorted(
            permission.value
            for permission
            in permissions_for_roles(
                normalized_roles
            )
        )
    )

    payload = TokenPayload(
        subject=str(subject),
        token_type=token_type,
        token_id=str(uuid4()),
        roles=normalized_roles,
        permissions=permissions,
        issued_at=now,
        expires_at=expires_at,
        issuer=security_config.issuer,
        audience=security_config.audience,
    )

    encoded = jwt.encode(
        payload.to_dict(),
        security_config.secret_key,
        algorithm=security_config.algorithm,
    )

    return encoded, payload


def create_access_token(
    *,
    subject: str,
    roles: Iterable[str | Role] = (),
) -> str:

    token, _ = _create_token(
        subject=subject,
        token_type=TokenType.ACCESS,
        roles=roles,
        expires_delta=timedelta(
            minutes=(
                security_config
                .access_token_minutes
            )
        ),
    )

    return token


def create_refresh_token(
    *,
    subject: str,
    roles: Iterable[str | Role] = (),
) -> str:

    token, _ = _create_token(
        subject=subject,
        token_type=TokenType.REFRESH,
        roles=roles,
        expires_delta=timedelta(
            days=(
                security_config
                .refresh_token_days
            )
        ),
    )

    return token


def create_token_pair(
    *,
    subject: str,
    roles: Iterable[str | Role],
) -> dict[str, Any]:

    access_token, access_payload = (
        _create_token(
            subject=subject,
            token_type=TokenType.ACCESS,
            roles=roles,
            expires_delta=timedelta(
                minutes=(
                    security_config
                    .access_token_minutes
                )
            ),
        )
    )

    refresh_token, refresh_payload = (
        _create_token(
            subject=subject,
            token_type=TokenType.REFRESH,
            roles=roles,
            expires_delta=timedelta(
                days=(
                    security_config
                    .refresh_token_days
                )
            ),
        )
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "access_expires_at": (
            access_payload.expires_at.isoformat()
        ),
        "refresh_expires_at": (
            refresh_payload.expires_at.isoformat()
        ),
    }


# =========================================================
# JWT verification
# =========================================================

def decode_token(
    token: str,
    *,
    expected_type: TokenType | None = None,
) -> dict[str, Any]:

    security_config.validate()

    try:
        payload = jwt.decode(
            token,
            security_config.secret_key,
            algorithms=[
                security_config.algorithm
            ],
            audience=(
                security_config.audience
            ),
            issuer=(
                security_config.issuer
            ),
            options={
                "require": [
                    "sub",
                    "type",
                    "jti",
                    "iat",
                    "exp",
                    "iss",
                    "aud",
                ],
            },
        )

    except ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="Token has expired.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        ) from exc

    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="Invalid authentication token.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        ) from exc

    if expected_type is not None:
        if payload.get("type") != (
            expected_type.value
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_401_UNAUTHORIZED
                ),
                detail=(
                    "Incorrect token type."
                ),
            )

    return payload


# =========================================================
# API keys
# =========================================================

def generate_api_key() -> str:
    """
    Return the plaintext key once.

    Store only hash_api_key(key) in the DB.
    """
    return (
        "iai_"
        + secrets.token_urlsafe(32)
    )


def hash_api_key(
    api_key: str,
) -> str:
    security_config.validate()

    return hmac.new(
        security_config.api_key_pepper.encode(),
        api_key.encode(),
        hashlib.sha256,
    ).hexdigest()


def verify_api_key(
    api_key: str,
    stored_hash: str,
) -> bool:

    candidate = hash_api_key(
        api_key
    )

    return hmac.compare_digest(
        candidate,
        stored_hash,
    )


# =========================================================
# General secure tokens
# =========================================================

def generate_secure_token(
    length: int = 32,
) -> str:
    if length < 16:
        raise ValueError(
            "Secure token length must "
            "be at least 16."
        )

    return secrets.token_urlsafe(
        length
    )


def hash_token(
    token: str,
) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


# =========================================================
# FastAPI authentication
# =========================================================

bearer_scheme = HTTPBearer(
    auto_error=False
)


async def get_token_payload(
    credentials: (
        HTTPAuthorizationCredentials | None
    ) = Depends(bearer_scheme),
) -> dict[str, Any]:

    if credentials is None:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="Authentication required.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    return decode_token(
        credentials.credentials,
        expected_type=TokenType.ACCESS,
    )


async def get_current_identity(
    payload: dict[str, Any] = Depends(
        get_token_payload
    ),
) -> dict[str, Any]:

    subject = payload.get("sub")

    if not subject:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="Invalid identity.",
        )

    return {
        "user_id": subject,
        "roles": payload.get(
            "roles",
            [],
        ),
        "permissions": payload.get(
            "permissions",
            [],
        ),
        "token_id": payload.get("jti"),
    }


# =========================================================
# Permission dependency
# =========================================================

class RequirePermission:

    def __init__(
        self,
        *permissions: Permission,
    ) -> None:
        self.permissions = {
            permission.value
            for permission in permissions
        }

    async def __call__(
        self,
        identity: dict[str, Any] = Depends(
            get_current_identity
        ),
    ) -> dict[str, Any]:

        granted = set(
            identity.get(
                "permissions",
                [],
            )
        )

        if Permission.ADMIN.value in granted:
            return identity

        missing = (
            self.permissions - granted
        )

        if missing:
            raise HTTPException(
                status_code=(
                    status.HTTP_403_FORBIDDEN
                ),
                detail={
                    "message": (
                        "Insufficient permissions."
                    ),
                    "missing": sorted(
                        missing
                    ),
                },
            )

        return identity


# =========================================================
# Role dependency
# =========================================================

class RequireRole:

    def __init__(
        self,
        *roles: Role,
    ) -> None:
        self.roles = {
            role.value
            for role in roles
        }

    async def __call__(
        self,
        identity: dict[str, Any] = Depends(
            get_current_identity
        ),
    ) -> dict[str, Any]:

        current_roles = set(
            identity.get(
                "roles",
                [],
            )
        )

        if Role.ADMIN.value in current_roles:
            return identity

        if not (
            self.roles & current_roles
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_403_FORBIDDEN
                ),
                detail="Required role missing.",
            )

        return identity


# =========================================================
# Security headers
# =========================================================

async def add_security_headers(
    request: Request,
    call_next,
):
    response = await call_next(
        request
    )

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "strict-origin-when-cross-origin"

    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), microphone=(), "
        "geolocation=()"
    )

    response.headers[
        "Content-Security-Policy"
    ] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "connect-src 'self' ws: wss:;"
    )

    if request.url.scheme == "https":
        response.headers[
            "Strict-Transport-Security"
        ] = (
            "max-age=31536000; "
            "includeSubDomains"
        )

    return response