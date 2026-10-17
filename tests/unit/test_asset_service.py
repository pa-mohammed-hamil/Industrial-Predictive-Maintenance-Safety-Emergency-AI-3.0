
"""
tests/unit/test_asset_service.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Asset Service Unit Tests

Run:
    pytest tests/unit/test_asset_service.py -v

Coverage:
    - Asset creation
    - Asset retrieval
    - Asset listing
    - Asset updates
    - Asset deletion
    - Duplicate asset prevention
    - Asset status management
    - Input validation
    - Tenant isolation
    - Repository failure handling
    - Data integrity

IMPORTANT:
This is a proposed unit-test contract.

Connect AssetServiceAdapter to the actual asset
service implementation in app/services/.

Tests use an in-memory fake repository. They do
not require a real database or industrial equipment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

import pytest


# ==================================================
# Domain models
# ==================================================

VALID_STATUSES = {
    "active",
    "inactive",
    "maintenance",
    "retired",
}


@dataclass
class Asset:
    id: str
    asset_id: str
    name: str
    asset_type: str
    site_id: str
    tenant_id: str
    status: str = "active"
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )
    updated_at: datetime | None = None


# ==================================================
# Custom exceptions
# ==================================================

class AssetNotFoundError(Exception):
    pass


class DuplicateAssetError(Exception):
    pass


class AssetValidationError(Exception):
    pass


class AssetAccessDeniedError(Exception):
    pass


# ==================================================
# Fake repository
# ==================================================

class FakeAssetRepository:
    """
    In-memory repository for isolated unit tests.

    This fake provides deterministic behavior
    without requiring PostgreSQL.
    """

    def __init__(self) -> None:
        self.assets: dict[str, Asset] = {}
        self.fail_next_operation = False

    def _check_failure(self) -> None:
        if self.fail_next_operation:
            self.fail_next_operation = False
            raise RuntimeError(
                "Simulated repository failure"
            )

    def create(self, asset: Asset) -> Asset:
        self._check_failure()

        for existing in self.assets.values():
            if (
                existing.asset_id == asset.asset_id
                and existing.tenant_id == asset.tenant_id
            ):
                raise DuplicateAssetError(
                    "Asset already exists"
                )

        self.assets[asset.id] = deepcopy(asset)
        return deepcopy(asset)

    def get(
        self,
        asset_id: str,
        tenant_id: str,
    ) -> Asset | None:
        self._check_failure()

        asset = self.assets.get(asset_id)

        if asset is None:
            return None

        if asset.tenant_id != tenant_id:
            return None

        return deepcopy(asset)

    def list(
        self,
        tenant_id: str,
    ) -> list[Asset]:
        self._check_failure()

        return [
            deepcopy(asset)
            for asset in self.assets.values()
            if asset.tenant_id == tenant_id
        ]

    def update(
        self,
        asset: Asset,
    ) -> Asset:
        self._check_failure()

        if asset.id not in self.assets:
            raise AssetNotFoundError(
                "Asset not found"
            )

        self.assets[asset.id] = deepcopy(asset)
        return deepcopy(asset)

    def delete(
        self,
        asset_id: str,
        tenant_id: str,
    ) -> bool:
        self._check_failure()

        asset = self.assets.get(asset_id)

        if asset is None:
            return False

        if asset.tenant_id != tenant_id:
            return False

        del self.assets[asset_id]
        return True


# ==================================================
# Service interface
# ==================================================

class AssetServiceProtocol(Protocol):

    def create_asset(
        self,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> Asset:
        ...

    def get_asset(
        self,
        asset_id: str,
        tenant_id: str,
    ) -> Asset:
        ...

    def list_assets(
        self,
        tenant_id: str,
    ) -> list[Asset]:
        ...

    def update_asset(
        self,
        asset_id: str,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> Asset:
        ...

    def delete_asset(
        self,
        asset_id: str,
        tenant_id: str,
    ) -> bool:
        ...


# ==================================================
# Production service adapter
# ==================================================

class AssetServiceAdapter:
    """
    Connect this adapter to your actual service.

    Example:

        from app.services.asset_service import AssetService

        self.service = AssetService(repository=repository)

    Then map the methods below to the actual
    service methods and return types.

    Do not implement the business logic inside
    this adapter if the purpose is to test the
    production AssetService.
    """

    def __init__(
        self,
        repository: FakeAssetRepository,
    ) -> None:
        self.repository = repository
        self.service: Any = None

        # TODO: Initialize the actual service.

    def create_asset(
        self,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> Asset:
        raise NotImplementedError(
            "Connect create_asset to production service"
        )

    def get_asset(
        self,
        asset_id: str,
        tenant_id: str,
    ) -> Asset:
        raise NotImplementedError(
            "Connect get_asset to production service"
        )

    def list_assets(
        self,
        tenant_id: str,
    ) -> list[Asset]:
        raise NotImplementedError(
            "Connect list_assets to production service"
        )

    def update_asset(
        self,
        asset_id: str,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> Asset:
        raise NotImplementedError(
            "Connect update_asset to production service"
        )

    def delete_asset(
        self,
        asset_id: str,
        tenant_id: str,
    ) -> bool:
        raise NotImplementedError(
            "Connect delete_asset to production service"
        )


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def repository() -> FakeAssetRepository:
    return FakeAssetRepository()


@pytest.fixture
def service(
    repository: FakeAssetRepository,
) -> AssetServiceProtocol:

    adapter = AssetServiceAdapter(repository)

    if adapter.service is None:
        pytest.skip(
            "Connect AssetServiceAdapter to the "
            "production asset service."
        )

    return adapter


@pytest.fixture
def asset_payload() -> dict[str, Any]:
    return {
        "asset_id": "PUMP-001",
        "name": "Industrial Water Pump",
        "asset_type": "pump",
        "site_id": "PLANT-001",
        "status": "active",
        "manufacturer": "Test Manufacturer",
        "model": "PUMP-X100",
        "serial_number": "SN-001",
    }


@pytest.fixture
def created_asset(
    service: AssetServiceProtocol,
    asset_payload: dict[str, Any],
) -> Asset:

    return service.create_asset(
        asset_payload,
        tenant_id="TENANT-A",
    )


# ==================================================
# Asset creation tests
# ==================================================

class TestAssetCreation:

    def test_create_asset(
        self,
        created_asset: Asset,
    ) -> None:

        assert created_asset.id
        assert created_asset.asset_id == "PUMP-001"
        assert created_asset.name == "Industrial Water Pump"
        assert created_asset.asset_type == "pump"
        assert created_asset.site_id == "PLANT-001"
        assert created_asset.tenant_id == "TENANT-A"

    def test_default_status(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        payload = deepcopy(asset_payload)
        payload.pop("status")

        result = service.create_asset(
            payload,
            "TENANT-A",
        )

        assert result.status == "active"

    def test_created_at_is_timezone_aware(
        self,
        created_asset: Asset,
    ) -> None:

        assert created_asset.created_at.tzinfo is not None

    def test_duplicate_asset_rejected(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        service.create_asset(
            asset_payload,
            "TENANT-A",
        )

        with pytest.raises(DuplicateAssetError):
            service.create_asset(
                asset_payload,
                "TENANT-A",
            )

    def test_same_asset_code_different_tenants(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        first = service.create_asset(
            asset_payload,
            "TENANT-A",
        )

        second = service.create_asset(
            asset_payload,
            "TENANT-B",
        )

        assert first.id != second.id
        assert first.tenant_id != second.tenant_id


# ==================================================
# Asset retrieval tests
# ==================================================

class TestAssetRetrieval:

    def test_get_existing_asset(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        result = service.get_asset(
            created_asset.id,
            "TENANT-A",
        )

        assert result.id == created_asset.id
        assert result.asset_id == created_asset.asset_id

    def test_get_unknown_asset(
        self,
        service: AssetServiceProtocol,
    ) -> None:

        with pytest.raises(AssetNotFoundError):
            service.get_asset(
                str(uuid4()),
                "TENANT-A",
            )

    def test_cross_tenant_access_denied(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        with pytest.raises(
            (AssetNotFoundError, AssetAccessDeniedError)
        ):
            service.get_asset(
                created_asset.id,
                "TENANT-B",
            )


# ==================================================
# Asset listing tests
# ==================================================

class TestAssetListing:

    def test_list_assets(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        for index in range(3):
            payload = {
                **asset_payload,
                "asset_id": f"PUMP-{index + 1:03d}",
            }

            service.create_asset(
                payload,
                "TENANT-A",
            )

        assets = service.list_assets(
            "TENANT-A"
        )

        assert len(assets) == 3

    def test_empty_asset_list(
        self,
        service: AssetServiceProtocol,
    ) -> None:

        result = service.list_assets(
            "TENANT-A"
        )

        assert result == []

    def test_list_tenant_isolation(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        service.create_asset(
            asset_payload,
            "TENANT-A",
        )

        service.create_asset(
            {
                **asset_payload,
                "asset_id": "PUMP-B-001",
            },
            "TENANT-B",
        )

        assets = service.list_assets(
            "TENANT-A"
        )

        assert len(assets) == 1

        assert all(
            asset.tenant_id == "TENANT-A"
            for asset in assets
        )


# ==================================================
# Asset update tests
# ==================================================

class TestAssetUpdates:

    def test_update_asset_name(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        result = service.update_asset(
            created_asset.id,
            {"name": "Updated Industrial Pump"},
            "TENANT-A",
        )

        assert result.name == "Updated Industrial Pump"

    def test_update_asset_status(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        result = service.update_asset(
            created_asset.id,
            {"status": "maintenance"},
            "TENANT-A",
        )

        assert result.status == "maintenance"

    def test_update_asset_model(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        result = service.update_asset(
            created_asset.id,
            {"model": "PUMP-X200"},
            "TENANT-A",
        )

        assert result.model == "PUMP-X200"

    def test_update_unknown_asset(
        self,
        service: AssetServiceProtocol,
    ) -> None:

        with pytest.raises(AssetNotFoundError):
            service.update_asset(
                str(uuid4()),
                {"name": "Unknown"},
                "TENANT-A",
            )

    def test_cross_tenant_update_denied(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        with pytest.raises(
            (AssetNotFoundError, AssetAccessDeniedError)
        ):
            service.update_asset(
                created_asset.id,
                {"name": "Unauthorized"},
                "TENANT-B",
            )


# ==================================================
# Asset deletion tests
# ==================================================

class TestAssetDeletion:

    def test_delete_asset(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        result = service.delete_asset(
            created_asset.id,
            "TENANT-A",
        )

        assert result is True

        with pytest.raises(AssetNotFoundError):
            service.get_asset(
                created_asset.id,
                "TENANT-A",
            )

    def test_delete_unknown_asset(
        self,
        service: AssetServiceProtocol,
    ) -> None:

        result = service.delete_asset(
            str(uuid4()),
            "TENANT-A",
        )

        assert result is False

    def test_cross_tenant_delete_denied(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        with pytest.raises(
            (AssetNotFoundError, AssetAccessDeniedError)
        ):
            service.delete_asset(
                created_asset.id,
                "TENANT-B",
            )


# ==================================================
# Validation tests
# ==================================================

class TestAssetValidation:

    @pytest.mark.parametrize(
        "missing_field",
        [
            "asset_id",
            "name",
            "asset_type",
            "site_id",
        ],
    )
    def test_missing_required_fields(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
        missing_field: str,
    ) -> None:

        payload = deepcopy(asset_payload)
        payload.pop(missing_field)

        with pytest.raises(
            (AssetValidationError, ValueError)
        ):
            service.create_asset(
                payload,
                "TENANT-A",
            )

    @pytest.mark.parametrize(
        "invalid_status",
        [
            "unknown",
            "broken-status",
            "",
        ],
    )
    def test_invalid_status(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
        invalid_status: str,
    ) -> None:

        payload = {
            **asset_payload,
            "status": invalid_status,
        }

        with pytest.raises(
            (AssetValidationError, ValueError)
        ):
            service.create_asset(
                payload,
                "TENANT-A",
            )

    def test_empty_asset_name(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        payload = {
            **asset_payload,
            "name": "",
        }

        with pytest.raises(
            (AssetValidationError, ValueError)
        ):
            service.create_asset(
                payload,
                "TENANT-A",
            )

    def test_invalid_tenant(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        with pytest.raises(
            (AssetValidationError, ValueError)
        ):
            service.create_asset(
                asset_payload,
                "",
            )


# ==================================================
# Data integrity tests
# ==================================================

class TestAssetDataIntegrity:

    def test_input_payload_not_mutated(
        self,
        service: AssetServiceProtocol,
        asset_payload: dict[str, Any],
    ) -> None:

        original = deepcopy(asset_payload)

        service.create_asset(
            asset_payload,
            "TENANT-A",
        )

        assert asset_payload == original

    def test_update_preserves_asset_identity(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        result = service.update_asset(
            created_asset.id,
            {"name": "Renamed Pump"},
            "TENANT-A",
        )

        assert result.id == created_asset.id
        assert result.asset_id == created_asset.asset_id
        assert result.tenant_id == created_asset.tenant_id

    def test_update_does_not_modify_other_fields(
        self,
        service: AssetServiceProtocol,
        created_asset: Asset,
    ) -> None:

        result = service.update_asset(
            created_asset.id,
            {"name": "Renamed Pump"},
            "TENANT-A",
        )

        assert result.asset_type == created_asset.asset_type
        assert result.site_id == created_asset.site_id
        assert result.manufacturer == created_asset.manufacturer


# ==================================================
# Repository failure tests
# ==================================================

class TestRepositoryFailures:

    def test_create_repository_failure(
        self,
        service: AssetServiceProtocol,
        repository: FakeAssetRepository,
        asset_payload: dict[str, Any],
    ) -> None:

        repository.fail_next_operation = True

        with pytest.raises(RuntimeError):
            service.create_asset(
                asset_payload,
                "TENANT-A",
            )

    def test_read_repository_failure(
        self,
        service: AssetServiceProtocol,
        repository: FakeAssetRepository,
        created_asset: Asset,
    ) -> None:

        repository.fail_next_operation = True

        with pytest.raises(RuntimeError):
            service.get_asset(
                created_asset.id,
                "TENANT-A",
            )

    def test_delete_repository_failure(
        self,
        service: AssetServiceProtocol,
        repository: FakeAssetRepository,
        created_asset: Asset,
    ) -> None:

        repository.fail_next_operation = True

        with pytest.raises(RuntimeError):
            service.delete_asset(
                created_asset.id,
                "TENANT-A",
            )
