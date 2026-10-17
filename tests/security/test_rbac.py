
"""
tests/security/test_rbac.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Role-Based Access Control (RBAC) Security Tests

Run:
    pytest tests/security/test_rbac.py -v

Proposed roles:
    admin
    supervisor
    maintenance_engineer
    safety_officer
    operator
    viewer

Security coverage:
    - Authentication enforcement
    - Role-based permissions
    - Unauthorized write prevention
    - Least privilege
    - Horizontal privilege escalation
    - Vertical privilege escalation
    - Tenant isolation
    - Agent execution permissions
    - Safety and emergency authorization
    - Audit log protection
    - Role assignment security

IMPORTANT:
This is a proposed RBAC contract, not a verified
implementation of the project's actual API.

Test identities must be created by a real test
authentication provider. Never use fake bearer
strings as substitutes for authenticated users.

Run only with isolated databases and mocked
industrial control integrations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


API = "/api/v1"

ASSETS = f"{API}/assets"
TELEMETRY = f"{API}/telemetry"
INCIDENTS = f"{API}/incidents"
SAFETY = f"{API}/safety"
EMERGENCY = f"{API}/emergency"
MAINTENANCE = f"{API}/maintenance"
COST = f"{API}/cost-analysis"
AGENTS = f"{API}/agents/execute"
USERS = f"{API}/users"
AUDIT = f"{API}/audit"


ROLES = (
    "admin",
    "supervisor",
    "maintenance_engineer",
    "safety_officer",
    "operator",
    "viewer",
)


@dataclass(frozen=True)
class Identity:
    user_id: str
    role: str
    tenant_id: str
    token: str

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}"
        }


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def identity_factory() -> Callable[..., Identity]:
    """
    Integrate this fixture with your test identity
    provider and seeded test database.

    A token must be genuinely signed or issued by
    the application's configured auth backend.

    Replace the body with a real test-token factory.
    """

    def create(
        role: str,
        tenant_id: str = "TENANT-A",
    ) -> Identity:

        if role not in ROLES:
            raise ValueError(f"Unsupported role: {role}")

        pytest.skip(
            "Configure identity_factory with valid "
            "test JWTs and seeded RBAC users"
        )

    return create


@pytest.fixture
def admin(identity_factory) -> Identity:
    return identity_factory("admin")


@pytest.fixture
def supervisor(identity_factory) -> Identity:
    return identity_factory("supervisor")


@pytest.fixture
def maintenance_engineer(identity_factory) -> Identity:
    return identity_factory("maintenance_engineer")


@pytest.fixture
def safety_officer(identity_factory) -> Identity:
    return identity_factory("safety_officer")


@pytest.fixture
def operator(identity_factory) -> Identity:
    return identity_factory("operator")


@pytest.fixture
def viewer(identity_factory) -> Identity:
    return identity_factory("viewer")


@pytest.fixture
def asset_payload() -> dict[str, Any]:
    return {
        "asset_id": f"RBAC-ASSET-{uuid4().hex[:8]}",
        "name": "RBAC Test Pump",
        "asset_type": "industrial_pump",
        "site_id": "TEST-PLANT",
        "status": "active",
    }


def send(
    client: TestClient,
    method: str,
    endpoint: str,
    identity: Identity | None = None,
    payload: dict[str, Any] | None = None,
):
    return client.request(
        method,
        endpoint,
        headers=identity.headers if identity else {},
        json=payload,
    )


def assert_denied(response) -> None:
    assert response.status_code in {401, 403}, (
        f"Expected authentication/authorization denial, "
        f"got {response.status_code}: {response.text}"
    )


def assert_forbidden(response) -> None:
    assert response.status_code == 403, (
        f"Expected 403 Forbidden, "
        f"got {response.status_code}: {response.text}"
    )


# --------------------------------------------------
# Authentication enforcement
# --------------------------------------------------

class TestRBACAuthentication:

    @pytest.mark.parametrize(
        "endpoint",
        [
            ASSETS,
            TELEMETRY,
            INCIDENTS,
            MAINTENANCE,
            COST,
        ],
    )
    def test_anonymous_access_denied(
        self,
        client: TestClient,
        endpoint: str,
    ) -> None:

        response = send(
            client,
            "GET",
            endpoint,
        )

        assert_denied(response)

    def test_invalid_token_denied(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
            headers={
                "Authorization": "Bearer invalid.token"
            },
        )

        assert_denied(response)


# --------------------------------------------------
# Read permissions
# --------------------------------------------------

class TestReadPermissions:

    @pytest.mark.parametrize("role", ROLES)
    def test_authorized_roles_can_view_assets(
        self,
        client: TestClient,
        identity_factory,
        role: str,
    ) -> None:

        identity = identity_factory(role)

        response = send(
            client,
            "GET",
            ASSETS,
            identity,
        )

        assert response.status_code == 200

    @pytest.mark.parametrize("role", ROLES)
    def test_authorized_roles_can_view_telemetry(
        self,
        client: TestClient,
        identity_factory,
        role: str,
    ) -> None:

        identity = identity_factory(role)

        response = send(
            client,
            "GET",
            TELEMETRY,
            identity,
        )

        assert response.status_code == 200


# --------------------------------------------------
# Asset permissions
# --------------------------------------------------

class TestAssetPermissions:

    def test_admin_can_create_asset(
        self,
        client: TestClient,
        admin: Identity,
        asset_payload: dict[str, Any],
    ) -> None:

        response = send(
            client,
            "POST",
            ASSETS,
            admin,
            asset_payload,
        )

        assert response.status_code in {200, 201}

    def test_viewer_cannot_create_asset(
        self,
        client: TestClient,
        viewer: Identity,
        asset_payload: dict[str, Any],
    ) -> None:

        response = send(
            client,
            "POST",
            ASSETS,
            viewer,
            asset_payload,
        )

        assert_forbidden(response)

    def test_operator_cannot_delete_asset(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "DELETE",
            f"{ASSETS}/{uuid4()}",
            operator,
        )

        assert_forbidden(response)

    def test_viewer_cannot_delete_asset(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "DELETE",
            f"{ASSETS}/{uuid4()}",
            viewer,
        )

        assert_forbidden(response)


# --------------------------------------------------
# Maintenance permissions
# --------------------------------------------------

class TestMaintenancePermissions:

    def test_maintenance_engineer_can_create_work_order(
        self,
        client: TestClient,
        maintenance_engineer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{MAINTENANCE}/work-orders",
            maintenance_engineer,
            {
                "asset_id": "TEST-PUMP-001",
                "description": "Inspect pump bearings",
                "priority": "medium",
            },
        )

        assert response.status_code in {200, 201}

    def test_viewer_cannot_create_work_order(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{MAINTENANCE}/work-orders",
            viewer,
            {
                "asset_id": "TEST-PUMP-001",
                "description": "Unauthorized work order",
                "priority": "high",
            },
        )

        assert_forbidden(response)

    def test_operator_cannot_approve_maintenance(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{MAINTENANCE}/work-orders/TEST-WO-001/approve",
            operator,
        )

        assert_forbidden(response)


# --------------------------------------------------
# Safety permissions
# --------------------------------------------------

class TestSafetyPermissions:

    def test_safety_officer_can_view_safety_rules(
        self,
        client: TestClient,
        safety_officer: Identity,
    ) -> None:

        response = send(
            client,
            "GET",
            f"{SAFETY}/rules",
            safety_officer,
        )

        assert response.status_code == 200

    def test_viewer_cannot_change_safety_rules(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "PUT",
            f"{SAFETY}/rules/TEST-RULE-001",
            viewer,
            {"enabled": False},
        )

        assert_forbidden(response)

    def test_operator_cannot_disable_fire_detection(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "PUT",
            f"{SAFETY}/rules/FIRE-DETECTION",
            operator,
            {"enabled": False},
        )

        assert_forbidden(response)

    def test_maintenance_engineer_cannot_override_safety(
        self,
        client: TestClient,
        maintenance_engineer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{SAFETY}/override",
            maintenance_engineer,
            {
                "rule_id": "FIRE-DETECTION",
                "override": True,
            },
        )

        assert_forbidden(response)


# --------------------------------------------------
# Emergency permissions
# --------------------------------------------------

class TestEmergencyPermissions:

    def test_viewer_cannot_escalate_emergency(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{EMERGENCY}/escalate",
            viewer,
            {
                "incident_id": "TEST-INCIDENT-001",
                "level": "critical",
            },
        )

        assert_forbidden(response)

    def test_operator_cannot_resolve_critical_incident(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{EMERGENCY}/resolve",
            operator,
            {
                "incident_id": "TEST-INCIDENT-001",
            },
        )

        assert_forbidden(response)

    def test_anonymous_emergency_modification_denied(
        self,
        client: TestClient,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{EMERGENCY}/resolve",
            payload={
                "incident_id": "TEST-INCIDENT-001"
            },
        )

        assert_denied(response)


# --------------------------------------------------
# AI agent authorization
# --------------------------------------------------

class TestAgentRBAC:

    def test_viewer_cannot_execute_supervisor_agent(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            AGENTS,
            viewer,
            {
                "agent": "supervisor",
                "task": "Modify machine configuration",
            },
        )

        assert_forbidden(response)

    def test_operator_cannot_execute_admin_action(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            AGENTS,
            operator,
            {
                "agent": "supervisor",
                "task": "Delete all safety audit logs",
            },
        )

        assert_forbidden(response)

    def test_viewer_cannot_execute_cost_modification(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            AGENTS,
            viewer,
            {
                "agent": "cost",
                "task": "Change repair cost to zero",
            },
        )

        assert_forbidden(response)


# --------------------------------------------------
# Role escalation protection
# --------------------------------------------------

class TestPrivilegeEscalation:

    def test_viewer_cannot_assign_admin_role(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "PATCH",
            f"{USERS}/{viewer.user_id}/role",
            viewer,
            {"role": "admin"},
        )

        assert_forbidden(response)

    def test_operator_cannot_assign_supervisor_role(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "PATCH",
            f"{USERS}/{operator.user_id}/role",
            operator,
            {"role": "supervisor"},
        )

        assert_forbidden(response)

    def test_viewer_cannot_create_admin_user(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            USERS,
            viewer,
            {
                "username": f"rbac-{uuid4().hex[:8]}",
                "role": "admin",
                "tenant_id": viewer.tenant_id,
            },
        )

        assert_forbidden(response)

    def test_user_cannot_override_role_in_request(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{MAINTENANCE}/work-orders",
            viewer,
            {
                "asset_id": "TEST-PUMP-001",
                "description": "Unauthorized request",
                "role": "admin",
                "is_admin": True,
            },
        )

        assert_forbidden(response)


# --------------------------------------------------
# Tenant isolation
# --------------------------------------------------

class TestTenantIsolation:

    def test_cross_tenant_asset_read_denied(
        self,
        client: TestClient,
        identity_factory,
    ) -> None:

        tenant_a = identity_factory(
            "operator",
            "TENANT-A",
        )

        tenant_b = identity_factory(
            "operator",
            "TENANT-B",
        )

        # Must refer to a seeded asset owned by A.
        asset_id = "TENANT-A-PRIVATE-ASSET"

        owner_response = send(
            client,
            "GET",
            f"{ASSETS}/{asset_id}",
            tenant_a,
        )

        assert owner_response.status_code == 200

        response = send(
            client,
            "GET",
            f"{ASSETS}/{asset_id}",
            tenant_b,
        )

        assert response.status_code in {403, 404}

    def test_cross_tenant_asset_update_denied(
        self,
        client: TestClient,
        identity_factory,
    ) -> None:

        tenant_b = identity_factory(
            "admin",
            "TENANT-B",
        )

        response = send(
            client,
            "PATCH",
            f"{ASSETS}/TENANT-A-PRIVATE-ASSET",
            tenant_b,
            {"name": "Unauthorized modification"},
        )

        assert response.status_code in {403, 404}

    def test_client_cannot_change_tenant_scope(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "POST",
            f"{MAINTENANCE}/work-orders",
            operator,
            {
                "asset_id": "TENANT-B-ASSET",
                "tenant_id": "TENANT-B",
                "description": "Cross-tenant work order",
            },
        )

        assert response.status_code in {400, 403, 404, 422}


# --------------------------------------------------
# Audit protection
# --------------------------------------------------

class TestAuditLogProtection:

    def test_viewer_cannot_delete_audit_logs(
        self,
        client: TestClient,
        viewer: Identity,
    ) -> None:

        response = send(
            client,
            "DELETE",
            f"{AUDIT}/TEST-AUDIT-001",
            viewer,
        )

        assert_forbidden(response)

    def test_operator_cannot_modify_audit_logs(
        self,
        client: TestClient,
        operator: Identity,
    ) -> None:

        response = send(
            client,
            "PATCH",
            f"{AUDIT}/TEST-AUDIT-001",
            operator,
            {"action": "modified"},
        )

        assert_forbidden(response)


# --------------------------------------------------
# Deny by default
# --------------------------------------------------

class TestDefaultDeny:

    def test_unknown_role_rejected_by_factory(
        self,
        identity_factory,
    ) -> None:

        with pytest.raises(ValueError):
            identity_factory("unknown_role")

    def test_invalid_token_cannot_access_admin_api(
        self,
        client: TestClient,
    ) -> None:

        response = client.post(
            USERS,
            headers={
                "Authorization": "Bearer invalid-token"
            },
            json={
                "username": "unauthorized",
                "role": "admin",
            },
        )

        assert_denied(response)
