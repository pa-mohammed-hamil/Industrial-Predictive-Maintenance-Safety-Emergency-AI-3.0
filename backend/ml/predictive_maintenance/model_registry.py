"""
Predictive Maintenance Model Registry
=====================================

Central registry for predictive-maintenance ML models.

Supports:
- model registration
- semantic/custom versions
- model stages
- active production versions
- model loading
- in-memory caching
- artifact checksums
- metadata
- metrics
- health state
- promotion
- rollback
- prediction routing
- local artifact persistence

Designed for:
    failure_prediction.py
    rul_prediction.py
    degradation_model.py
    maintenance_forecasting.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# ==============================================================
# Exceptions
# ==============================================================


class ModelRegistryError(Exception):
    """Base registry exception."""


class ModelNotFoundError(ModelRegistryError):
    """Requested model/version does not exist."""


class ModelAlreadyExistsError(ModelRegistryError):
    """Model version is already registered."""


class ModelArtifactError(ModelRegistryError):
    """Model artifact is missing or corrupted."""


# ==============================================================
# Enums
# ==============================================================


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelHealth(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ==============================================================
# Registry record
# ==============================================================


@dataclass
class ModelRecord:
    name: str
    version: str
    model_type: str

    artifact_path: str

    stage: ModelStage = ModelStage.DEVELOPMENT
    health: ModelHealth = ModelHealth.UNKNOWN

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    checksum: Optional[str] = None

    framework: Optional[str] = None
    description: Optional[str] = None

    metrics: Dict[str, float] = field(
        default_factory=dict
    )

    parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    tags: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:

        result = asdict(self)

        result["stage"] = self.stage.value
        result["health"] = self.health.value

        return result

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ModelRecord":

        payload = dict(data)

        payload["stage"] = ModelStage(
            payload.get(
                "stage",
                ModelStage.DEVELOPMENT.value,
            )
        )

        payload["health"] = ModelHealth(
            payload.get(
                "health",
                ModelHealth.UNKNOWN.value,
            )
        )

        return cls(**payload)


# ==============================================================
# Registry
# ==============================================================


class ModelRegistry:
    """
    Thread-safe local model registry.

    Registry layout:

        models/
        ├── registry.json
        ├── failure_prediction/
        │   └── 1.0.0/
        │       └── model.pkl
        ├── rul/
        ├── degradation/
        └── maintenance_forecasting/
    """

    def __init__(
        self,
        root_dir: str | Path = "models",
        registry_filename: str = "registry.json",
        verify_checksum_on_load: bool = True,
    ) -> None:

        self.root_dir = Path(
            root_dir
        ).resolve()

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.registry_path = (
            self.root_dir
            / registry_filename
        )

        self.verify_checksum_on_load = (
            verify_checksum_on_load
        )

        self._lock = threading.RLock()

        self._records: Dict[
            str,
            Dict[str, ModelRecord],
        ] = {}

        self._cache: Dict[
            tuple[str, str],
            Any,
        ] = {}

        self._load_registry()

    # ==========================================================
    # Registration
    # ==========================================================

    def register(
        self,
        *,
        name: str,
        version: str,
        model: Any,
        model_type: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        framework: Optional[str] = None,
        description: Optional[str] = None,
        metrics: Optional[Mapping[str, float]] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> ModelRecord:

        self._validate_identity(
            name,
            version,
        )

        with self._lock:

            existing = self._records.get(
                name,
                {},
            ).get(version)

            if existing and not overwrite:
                raise ModelAlreadyExistsError(
                    f"{name}:{version} already exists."
                )

            directory = (
                self.root_dir
                / self._safe_name(name)
                / self._safe_name(version)
            )

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            artifact = (
                directory
                / "model.pkl"
            )

            self._save_pickle(
                model,
                artifact,
            )

            checksum = self._checksum(
                artifact
            )

            now = datetime.now(
                timezone.utc
            ).isoformat()

            record = ModelRecord(
                name=name,
                version=version,
                model_type=model_type,
                artifact_path=str(
                    artifact
                ),
                stage=stage,
                health=ModelHealth.UNKNOWN,
                created_at=(
                    existing.created_at
                    if existing
                    else now
                ),
                updated_at=now,
                checksum=checksum,
                framework=framework,
                description=description,
                metrics=dict(
                    metrics or {}
                ),
                parameters=dict(
                    parameters or {}
                ),
                metadata=dict(
                    metadata or {}
                ),
                tags=list(
                    tags or []
                ),
            )

            self._records.setdefault(
                name,
                {},
            )[version] = record

            self._cache[
                (name, version)
            ] = model

            # Keep only one production version.
            if stage == ModelStage.PRODUCTION:
                self._demote_other_production_versions(
                    name=name,
                    keep_version=version,
                )

            self._persist()

            return record

    # ==========================================================
    # Register existing artifact
    # ==========================================================

    def register_artifact(
        self,
        *,
        name: str,
        version: str,
        artifact_path: str | Path,
        model_type: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        framework: Optional[str] = None,
        metrics: Optional[Mapping[str, float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ModelRecord:

        path = Path(
            artifact_path
        ).resolve()

        if not path.exists():
            raise ModelArtifactError(
                f"Artifact does not exist: {path}"
            )

        self._validate_identity(
            name,
            version,
        )

        record = ModelRecord(
            name=name,
            version=version,
            model_type=model_type,
            artifact_path=str(path),
            stage=stage,
            checksum=self._checksum(path),
            framework=framework,
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
        )

        with self._lock:

            if (
                name in self._records
                and version in self._records[name]
            ):
                raise ModelAlreadyExistsError(
                    f"{name}:{version} already exists."
                )

            self._records.setdefault(
                name,
                {},
            )[version] = record

            if stage == ModelStage.PRODUCTION:
                self._demote_other_production_versions(
                    name,
                    version,
                )

            self._persist()

        return record

    # ==========================================================
    # Loading
    # ==========================================================

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        *,
        stage: Optional[ModelStage] = None,
        use_cache: bool = True,
    ) -> Any:

        record = self.resolve(
            name=name,
            version=version,
            stage=stage,
        )

        key = (
            record.name,
            record.version,
        )

        with self._lock:

            if (
                use_cache
                and key in self._cache
            ):
                return self._cache[key]

        path = Path(
            record.artifact_path
        )

        self._validate_artifact(
            record,
            path,
        )

        model = self._load_pickle(
            path
        )

        if use_cache:
            with self._lock:
                self._cache[key] = model

        return model

    # ==========================================================
    # Resolution
    # ==========================================================

    def resolve(
        self,
        *,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
    ) -> ModelRecord:

        versions = self._records.get(
            name
        )

        if not versions:
            raise ModelNotFoundError(
                f"Model '{name}' not found."
            )

        if version is not None:

            record = versions.get(
                version
            )

            if record is None:
                raise ModelNotFoundError(
                    f"Model '{name}:{version}' not found."
                )

            return record

        desired_stage = (
            stage
            or ModelStage.PRODUCTION
        )

        candidates = [
            record
            for record in versions.values()
            if record.stage == desired_stage
        ]

        if not candidates:
            raise ModelNotFoundError(
                f"No {desired_stage.value} version "
                f"registered for '{name}'."
            )

        return max(
            candidates,
            key=lambda record: (
                self._version_key(
                    record.version
                ),
                record.updated_at,
            ),
        )

    # ==========================================================
    # Prediction routing
    # ==========================================================

    def predict(
        self,
        name: str,
        data: Any,
        *,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        method: str = "predict",
        **kwargs: Any,
    ) -> Any:

        model = self.load(
            name=name,
            version=version,
            stage=stage,
        )

        prediction_method = getattr(
            model,
            method,
            None,
        )

        if prediction_method is None:
            raise ModelRegistryError(
                f"Model '{name}' has no "
                f"'{method}' method."
            )

        return prediction_method(
            data,
            **kwargs,
        )

    # ==========================================================
    # Promotion
    # ==========================================================

    def promote(
        self,
        name: str,
        version: str,
        stage: ModelStage,
    ) -> ModelRecord:

        with self._lock:

            record = self.resolve(
                name=name,
                version=version,
            )

            if stage == ModelStage.PRODUCTION:
                self._demote_other_production_versions(
                    name,
                    version,
                )

            record.stage = stage

            record.updated_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            self._persist()

            return record

    # ==========================================================
    # Rollback
    # ==========================================================

    def rollback(
        self,
        name: str,
        *,
        target_version: Optional[str] = None,
    ) -> ModelRecord:

        with self._lock:

            versions = self._records.get(
                name
            )

            if not versions:
                raise ModelNotFoundError(
                    f"Model '{name}' not found."
                )

            current = [
                record
                for record in versions.values()
                if record.stage
                == ModelStage.PRODUCTION
            ]

            current_version = (
                current[0].version
                if current
                else None
            )

            if target_version is not None:

                target = versions.get(
                    target_version
                )

                if target is None:
                    raise ModelNotFoundError(
                        f"{name}:{target_version} "
                        "not found."
                    )

            else:

                candidates = [
                    record
                    for record in versions.values()
                    if (
                        record.version
                        != current_version
                        and record.stage
                        != ModelStage.ARCHIVED
                    )
                ]

                if not candidates:
                    raise ModelRegistryError(
                        "No rollback candidate available."
                    )

                target = max(
                    candidates,
                    key=lambda record:
                        self._version_key(
                            record.version
                        ),
                )

            if current_version:

                versions[
                    current_version
                ].stage = ModelStage.STAGING

            target.stage = (
                ModelStage.PRODUCTION
            )

            target.updated_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            self._persist()

            return target

    # ==========================================================
    # Health
    # ==========================================================

    def set_health(
        self,
        name: str,
        version: str,
        health: ModelHealth,
    ) -> ModelRecord:

        with self._lock:

            record = self.resolve(
                name=name,
                version=version,
            )

            record.health = health

            record.updated_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            self._persist()

            return record

    # ==========================================================
    # Metrics
    # ==========================================================

    def update_metrics(
        self,
        name: str,
        version: str,
        metrics: Mapping[str, float],
    ) -> ModelRecord:

        with self._lock:

            record = self.resolve(
                name=name,
                version=version,
            )

            record.metrics.update(
                {
                    str(key): float(value)
                    for key, value
                    in metrics.items()
                }
            )

            record.updated_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            self._persist()

            return record

    # ==========================================================
    # Metadata
    # ==========================================================

    def update_metadata(
        self,
        name: str,
        version: str,
        metadata: Mapping[str, Any],
    ) -> ModelRecord:

        with self._lock:

            record = self.resolve(
                name=name,
                version=version,
            )

            record.metadata.update(
                dict(metadata)
            )

            record.updated_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            self._persist()

            return record

    # ==========================================================
    # Listing
    # ==========================================================

    def list_models(
        self,
        *,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelRecord]:

        records: List[
            ModelRecord
        ] = []

        for versions in self._records.values():

            for record in versions.values():

                if (
                    stage is None
                    or record.stage == stage
                ):
                    records.append(
                        record
                    )

        return sorted(
            records,
            key=lambda record: (
                record.name,
                self._version_key(
                    record.version
                ),
            ),
        )

    def list_versions(
        self,
        name: str,
    ) -> List[ModelRecord]:

        versions = self._records.get(
            name
        )

        if not versions:
            return []

        return sorted(
            versions.values(),
            key=lambda record:
                self._version_key(
                    record.version
                ),
            reverse=True,
        )

    # ==========================================================
    # Production model
    # ==========================================================

    def get_production_model(
        self,
        name: str,
    ) -> ModelRecord:

        return self.resolve(
            name=name,
            stage=ModelStage.PRODUCTION,
        )

    # ==========================================================
    # Archive
    # ==========================================================

    def archive(
        self,
        name: str,
        version: str,
    ) -> ModelRecord:

        return self.promote(
            name=name,
            version=version,
            stage=ModelStage.ARCHIVED,
        )

    # ==========================================================
    # Cache
    # ==========================================================

    def clear_cache(
        self,
        name: Optional[str] = None,
    ) -> None:

        with self._lock:

            if name is None:
                self._cache.clear()
                return

            keys = [
                key
                for key in self._cache
                if key[0] == name
            ]

            for key in keys:
                self._cache.pop(
                    key,
                    None,
                )

    # ==========================================================
    # Registry persistence
    # ==========================================================

    def _load_registry(
        self,
    ) -> None:

        if not self.registry_path.exists():
            return

        try:

            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                payload = json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:

            raise ModelRegistryError(
                "Unable to read model registry."
            ) from exc

        models = payload.get(
            "models",
            {}
        )

        for name, versions in models.items():

            self._records[name] = {}

            for version, data in versions.items():

                self._records[
                    name
                ][version] = (
                    ModelRecord.from_dict(
                        data
                    )
                )

    def _persist(
        self,
    ) -> None:

        payload = {
            "schema_version": 1,
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "models": {
                name: {
                    version:
                        record.to_dict()
                    for version, record
                    in versions.items()
                }
                for name, versions
                in self._records.items()
            },
        }

        temporary = (
            self.registry_path
            .with_suffix(".tmp")
        )

        try:

            with temporary.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    payload,
                    file,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )

            temporary.replace(
                self.registry_path
            )

        except OSError as exc:

            raise ModelRegistryError(
                "Unable to persist registry."
            ) from exc

    # ==========================================================
    # Artifact handling
    # ==============================================================

    @staticmethod
    def _save_pickle(
        model: Any,
        path: Path,
    ) -> None:

        try:

            with path.open(
                "wb"
            ) as file:

                pickle.dump(
                    model,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        except Exception as exc:

            raise ModelArtifactError(
                f"Unable to save model: {path}"
            ) from exc

    @staticmethod
    def _load_pickle(
        path: Path,
    ) -> Any:

        try:

            with path.open(
                "rb"
            ) as file:

                return pickle.load(
                    file
                )

        except Exception as exc:

            raise ModelArtifactError(
                f"Unable to load model: {path}"
            ) from exc

    def _validate_artifact(
        self,
        record: ModelRecord,
        path: Path,
    ) -> None:

        if not path.exists():

            raise ModelArtifactError(
                f"Artifact missing: {path}"
            )

        if (
            self.verify_checksum_on_load
            and record.checksum
        ):

            checksum = self._checksum(
                path
            )

            if checksum != record.checksum:

                raise ModelArtifactError(
                    f"Checksum verification failed "
                    f"for {record.name}:"
                    f"{record.version}."
                )

    @staticmethod
    def _checksum(
        path: Path,
    ) -> str:

        digest = hashlib.sha256()

        with path.open(
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                digest.update(
                    chunk
                )

        return digest.hexdigest()

    # ==========================================================
    # Internal helpers
    # ==============================================================

    def _demote_other_production_versions(
        self,
        name: str,
        keep_version: str,
    ) -> None:

        versions = self._records.get(
            name,
            {},
        )

        for version, record in versions.items():

            if (
                version != keep_version
                and record.stage
                == ModelStage.PRODUCTION
            ):

                record.stage = (
                    ModelStage.STAGING
                )

                record.updated_at = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

    @staticmethod
    def _validate_identity(
        name: str,
        version: str,
    ) -> None:

        if not name.strip():
            raise ValueError(
                "Model name cannot be empty."
            )

        if not version.strip():
            raise ValueError(
                "Model version cannot be empty."
            )

    @staticmethod
    def _safe_name(
        value: str,
    ) -> str:

        safe = "".join(
            character
            if (
                character.isalnum()
                or character in "-_."
            )
            else "_"
            for character in value
        )

        if not safe:
            raise ValueError(
                "Invalid model identifier."
            )

        return safe

    @staticmethod
    def _version_key(
        version: str,
    ) -> tuple:

        """
        Provides predictable ordering for versions such as:
        1.0.0
        1.2.0
        2.0.0

        Also tolerates custom strings.
        """

        parts = version.replace(
            "-",
            ".",
        ).split(".")

        result = []

        for part in parts:

            if part.isdigit():
                result.append(
                    (1, int(part))
                )
            else:
                result.append(
                    (0, part)
                )

        return tuple(result)


# ==============================================================
# Default registry
# ==============================================================


_default_registry: Optional[
    ModelRegistry
] = None

_default_registry_lock = (
    threading.Lock()
)


def get_model_registry(
    root_dir: str | Path = "models",
) -> ModelRegistry:

    global _default_registry

    if _default_registry is None:

        with _default_registry_lock:

            if _default_registry is None:

                _default_registry = (
                    ModelRegistry(
                        root_dir=root_dir
                    )
                )

    return _default_registry


# ==============================================================
# Convenience functions
# ==============================================================


def register_model(
    *,
    name: str,
    version: str,
    model: Any,
    model_type: str,
    stage: ModelStage = ModelStage.DEVELOPMENT,
    metrics: Optional[
        Mapping[str, float]
    ] = None,
) -> ModelRecord:

    registry = get_model_registry()

    return registry.register(
        name=name,
        version=version,
        model=model,
        model_type=model_type,
        stage=stage,
        metrics=metrics,
    )


def load_model(
    name: str,
    *,
    version: Optional[str] = None,
) -> Any:

    registry = get_model_registry()

    return registry.load(
        name=name,
        version=version,
    )


# ==============================================================
# Demonstration
# ==============================================================


if __name__ == "__main__":

    from pprint import pprint

    # Example only. In the actual project import:
    #
    # from .failure_prediction import (
    #     create_default_failure_model
    # )
    #
    # model = create_default_failure_model()

    class DemoFailureModel:

        def predict(
            self,
            data: Any,
        ) -> Dict[str, Any]:

            return {
                "failure_probability": 0.72,
                "input": data,
            }

    registry = ModelRegistry(
        root_dir="models"
    )

    demo_model = DemoFailureModel()

    try:

        record = registry.register(
            name="failure_prediction",
            version="1.0.0",
            model=demo_model,
            model_type="failure_classifier",
            framework="python",
            stage=ModelStage.PRODUCTION,
            metrics={
                "roc_auc": 0.91,
                "f1": 0.86,
                "precision": 0.88,
                "recall": 0.84,
            },
            metadata={
                "dataset_version": "2026-10",
                "feature_schema": "v1",
            },
            tags=[
                "predictive-maintenance",
                "failure",
            ],
        )

        pprint(
            record.to_dict()
        )

    except ModelAlreadyExistsError:
        pass

    production = (
        registry.get_production_model(
            "failure_prediction"
        )
    )

    print(
        "Production:",
        production.version,
    )

    prediction = registry.predict(
        "failure_prediction",
        {
            "temperature": 92.0,
            "vibration_rms": 8.7,
        },
    )

    pprint(prediction)