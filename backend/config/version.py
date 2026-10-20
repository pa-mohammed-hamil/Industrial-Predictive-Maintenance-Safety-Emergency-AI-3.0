# app/config/version.py

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from typing import Any


# =========================================================
# Core versions
# =========================================================

APP_NAME = (
    "Industrial Predictive Maintenance "
    "Safety & Emergency Intelligence AI"
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)

API_VERSION = os.getenv(
    "API_VERSION",
    "v1",
)

API_PREFIX = f"/api/{API_VERSION}"

SCHEMA_VERSION = os.getenv(
    "SCHEMA_VERSION",
    "1.0",
)

EVENT_SCHEMA_VERSION = os.getenv(
    "EVENT_SCHEMA_VERSION",
    "1.0",
)

MODEL_REGISTRY_VERSION = os.getenv(
    "MODEL_REGISTRY_VERSION",
    "1.0",
)


# =========================================================
# Build metadata
# =========================================================

BUILD_NUMBER = os.getenv(
    "BUILD_NUMBER",
    "dev",
)

BUILD_ID = os.getenv(
    "BUILD_ID",
    BUILD_NUMBER,
)

GIT_COMMIT = os.getenv(
    "GIT_COMMIT",
    "unknown",
)

GIT_BRANCH = os.getenv(
    "GIT_BRANCH",
    "unknown",
)

BUILD_TIMESTAMP = os.getenv(
    "BUILD_TIMESTAMP",
    "unknown",
)

ENVIRONMENT = os.getenv(
    "APP_ENV",
    "development",
)


# =========================================================
# Version model
# =========================================================

@dataclass(
    frozen=True,
    slots=True,
)
class VersionInfo:
    application: str
    version: str

    api_version: str
    api_prefix: str

    schema_version: str
    event_schema_version: str
    model_registry_version: str

    environment: str

    build_number: str
    build_id: str

    git_commit: str
    git_branch: str

    build_timestamp: str

    python_version: str
    platform: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# Runtime package version
# =========================================================

def get_package_version(
    package_name: str,
) -> str | None:
    """
    Return installed package version.

    Example:
        get_package_version("fastapi")
    """

    try:
        return metadata.version(
            package_name
        )
    except metadata.PackageNotFoundError:
        return None


# =========================================================
# Runtime information
# =========================================================

def get_runtime_versions() -> dict[str, Any]:
    packages = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "asyncpg",
        "redis",
        "aiokafka",
        "pydantic",
        "pydantic-settings",
    ]

    installed = {}

    for package in packages:
        version = get_package_version(
            package
        )

        if version is not None:
            installed[package] = version

    return {
        "python": (
            platform.python_version()
        ),
        "implementation": (
            platform.python_implementation()
        ),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": installed,
    }


# =========================================================
# Version information
# =========================================================

def get_version_info() -> VersionInfo:
    return VersionInfo(
        application=APP_NAME,
        version=APP_VERSION,
        api_version=API_VERSION,
        api_prefix=API_PREFIX,
        schema_version=SCHEMA_VERSION,
        event_schema_version=(
            EVENT_SCHEMA_VERSION
        ),
        model_registry_version=(
            MODEL_REGISTRY_VERSION
        ),
        environment=ENVIRONMENT,
        build_number=BUILD_NUMBER,
        build_id=BUILD_ID,
        git_commit=GIT_COMMIT,
        git_branch=GIT_BRANCH,
        build_timestamp=BUILD_TIMESTAMP,
        python_version=(
            platform.python_version()
        ),
        platform=platform.platform(),
    )


# =========================================================
# Compact version
# =========================================================

def get_version() -> str:
    return APP_VERSION


def get_api_version() -> str:
    return API_VERSION


def get_api_prefix() -> str:
    return API_PREFIX


# =========================================================
# Build information
# =========================================================

def get_build_info() -> dict[str, Any]:
    return {
        "version": APP_VERSION,
        "build_number": BUILD_NUMBER,
        "build_id": BUILD_ID,
        "commit": GIT_COMMIT,
        "branch": GIT_BRANCH,
        "timestamp": BUILD_TIMESTAMP,
        "environment": ENVIRONMENT,
    }


# =========================================================
# Full system information
# =========================================================

def get_system_version_info() -> dict[str, Any]:
    return {
        "application": (
            get_version_info().to_dict()
        ),
        "runtime": (
            get_runtime_versions()
        ),
    }


# =========================================================
# Semantic version parsing
# =========================================================

def parse_version(
    version: str,
) -> tuple[int, int, int]:
    """
    Convert:
        1.2.3

    into:
        (1, 2, 3)
    """

    clean_version = (
        version
        .split("+", 1)[0]
        .split("-", 1)[0]
    )

    parts = clean_version.split(".")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid semantic version: "
            f"{version}"
        )

    try:
        return tuple(
            int(part)
            for part in parts
        )  # type: ignore[return-value]

    except ValueError as exc:
        raise ValueError(
            f"Invalid semantic version: "
            f"{version}"
        ) from exc


# =========================================================
# Version comparison
# =========================================================

def is_version_at_least(
    current: str,
    required: str,
) -> bool:
    return (
        parse_version(current)
        >= parse_version(required)
    )


def is_api_version_supported(
    version: str,
) -> bool:
    return version in {
        "v1",
    }


# =========================================================
# Event metadata
# =========================================================

def versioned_event_metadata() -> dict[str, str]:
    """
    Add this metadata to Kafka/MQTT events.
    """

    return {
        "application_version": APP_VERSION,
        "api_version": API_VERSION,
        "schema_version": (
            EVENT_SCHEMA_VERSION
        ),
    }


# =========================================================
# Model metadata
# =========================================================

def model_version_metadata(
    *,
    model_name: str,
    model_version: str,
) -> dict[str, str]:
    return {
        "model_name": model_name,
        "model_version": model_version,
        "registry_version": (
            MODEL_REGISTRY_VERSION
        ),
        "application_version": (
            APP_VERSION
        ),
    }


# =========================================================
# Startup banner
# =========================================================

def startup_banner() -> str:
    return (
        "\n"
        "====================================================\n"
        f"{APP_NAME}\n"
        "====================================================\n"
        f"Application : {APP_VERSION}\n"
        f"API         : {API_VERSION}\n"
        f"Environment : {ENVIRONMENT}\n"
        f"Build       : {BUILD_NUMBER}\n"
        f"Git Commit  : {GIT_COMMIT}\n"
        f"Python      : {platform.python_version()}\n"
        "===================================================="
    )


# =========================================================
# Health metadata
# =========================================================

def health_version_metadata() -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "api_version": API_VERSION,
        "environment": ENVIRONMENT,
        "build": BUILD_NUMBER,
        "commit": GIT_COMMIT,
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


__version__ = APP_VERSION


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "API_VERSION",
    "API_PREFIX",
    "SCHEMA_VERSION",
    "EVENT_SCHEMA_VERSION",
    "MODEL_REGISTRY_VERSION",
    "BUILD_NUMBER",
    "BUILD_ID",
    "GIT_COMMIT",
    "GIT_BRANCH",
    "BUILD_TIMESTAMP",
    "ENVIRONMENT",
    "VersionInfo",
    "get_version",
    "get_api_version",
    "get_api_prefix",
    "get_version_info",
    "get_build_info",
    "get_runtime_versions",
    "get_system_version_info",
    "parse_version",
    "is_version_at_least",
    "is_api_version_supported",
    "versioned_event_metadata",
    "model_version_metadata",
    "startup_banner",
    "health_version_metadata",
    "__version__",
]