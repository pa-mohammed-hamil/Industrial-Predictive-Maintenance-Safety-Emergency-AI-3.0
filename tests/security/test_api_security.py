
"""
tests/security/test_api_security.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

API Security Test Suite

Run:
    pytest tests/security/test_api_security.py -v

Dependencies:
    pytest
    httpx
    fastapi

IMPORTANT:
This is a proposed API security contract based on
the project's directory structure.

Adapt route paths, authentication configuration,
response schemas and test fixtures to your backend.

Run only against an isolated test environment.
Never connect these tests to production PLCs,
emergency dispatch, alarms or industrial actuators.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


API = "/api/v1"

AUTH = f"{API}/auth"
USERS = f"{API}/users"
ASSETS = f"{API}/assets"
TELEMETRY = f"{API}/telemetry"
SAFETY = f"{API}/safety"
INCIDENTS = f"{API}/incidents"
EMERGENCY = f"{API}/emergency"
COST = f"{API}/cost-analysis"
REPORTS = f"{API}/reports"


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin_headers() -> dict[str, str]:
    """
    Replace with a JWT issued by your isolated
    test authentication provider.
    """
    return {
        "Authorization": "Bearer TEST_ADMIN_TOKEN"
    }


@pytest.fixture
def operator_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer TEST_OPERATOR_TOKEN"
    }


@pytest.fixture
def viewer_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer TEST_VIEWER_TOKEN"
    }


@pytest.fixture
def asset_payload() -> dict[str, Any]:
    return {
        "asset_id": f"TEST-{uuid4().hex[:8]}",
        "asset_type": "industrial_pump",
        "site_id": "TEST-PLANT-001",
        "name": "Test Industrial Pump",
        "status": "active",
    }


@pytest.fixture
def telemetry_payload() -> dict[str, Any]:
    return {
        "asset_id": "TEST-MACHINE-001",
        "sensor_id": "TEST-SENSOR-001",
        "temperature": 45.0,
        "vibration": 2.5,
        "pressure": 4.2,
        "current": 12.0,
    }


# --------------------------------------------------
# Authentication
# --------------------------------------------------

class TestAuthentication:

    @pytest.mark.parametrize(
        "endpoint",
        [
            ASSETS,
            TELEMETRY,
            INCIDENTS,
            REPORTS,
        ],
    )
    def test_protected_endpoint_requires_auth(
        self,
        client: TestClient,
        endpoint: str,
    ) -> None:

        response = client.get(endpoint)

        assert response.status_code in {401, 403}

    def test_invalid_bearer_token(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
            headers={
                "Authorization": "Bearer invalid.token.value"
            },
        )

        assert response.status_code == 401

    def test_missing_bearer_token(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
            headers={
                "Authorization": "Bearer "
            },
        )

        assert response.status_code in {401, 403}

    def test_malformed_authorization_header(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
            headers={
                "Authorization": "InvalidScheme token"
            },
        )

        assert response.status_code in {401, 403}

    def test_random_token_rejected(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
            headers={
                "Authorization": f"Bearer {uuid4().hex}"
            },
        )

        assert response.status_code == 401


# --------------------------------------------------
# Authorization / RBAC
# --------------------------------------------------

class TestRoleBasedAccess:

    def test_admin_can_access_assets(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
    ) -> None:

        response = client.get(
            ASSETS,
            headers=admin_headers,
        )

        assert response.status_code == 200

    def test_viewer_cannot_create_asset(
        self,
        client: TestClient,
        viewer_headers: dict[str, str],
        asset_payload: dict[str, Any],
    ) -> None:

        response = client.post(
            ASSETS,
            headers=viewer_headers,
            json=asset_payload,
        )

        assert response.status_code == 403

    def test_operator_cannot_delete_asset(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = client.delete(
            f"{ASSETS}/{uuid4()}",
            headers=operator_headers,
        )

        assert response.status_code == 403

    def test_viewer_cannot_modify_safety_rules(
        self,
        client: TestClient,
        viewer_headers: dict[str, str],
    ) -> None:

        response = client.put(
            f"{SAFETY}/rules",
            headers=viewer_headers,
            json={
                "rule_id": "TEST-RULE-001",
                "enabled": False,
            },
        )

        assert response.status_code == 403


# --------------------------------------------------
# Tenant Isolation
# --------------------------------------------------

class TestTenantIsolation:

    def test_cross_tenant_asset_access_denied(
        self,
        client: TestClient,
    ) -> None:
        """
        These tokens must be generated by the
        test identity provider and mapped to
        separate seeded tenants.
        """

        tenant_a_headers = {
            "Authorization": "Bearer TEST_TENANT_A_TOKEN"
        }

        tenant_b_headers = {
            "Authorization": "Bearer TEST_TENANT_B_TOKEN"
        }

        asset_id = "TENANT-A-PRIVATE-ASSET"

        owner_response = client.get(
            f"{ASSETS}/{asset_id}",
            headers=tenant_a_headers,
        )

        assert owner_response.status_code == 200

        other_response = client.get(
            f"{ASSETS}/{asset_id}",
            headers=tenant_b_headers,
        )

        assert other_response.status_code in {403, 404}

    def test_tenant_id_cannot_be_overridden(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
        asset_payload: dict[str, Any],
    ) -> None:

        payload = {
            **asset_payload,
            "tenant_id": "OTHER-TENANT",
        }

        response = client.post(
            ASSETS,
            headers=operator_headers,
            json=payload,
        )

        assert response.status_code in {400, 403, 422}


# --------------------------------------------------
# Injection Protection
# --------------------------------------------------

class TestInjectionProtection:

    @pytest.mark.parametrize(
        "payload",
        [
            "' OR '1'='1",
            "'; DROP TABLE assets; --",
            "' UNION SELECT NULL --",
            "1 OR 1=1",
        ],
    )
    def test_sql_injection_does_not_bypass_auth(
        self,
        client: TestClient,
        payload: str,
    ) -> None:

        response = client.get(
            ASSETS,
            params={"search": payload},
        )

        assert response.status_code in {401, 403}

    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
        ],
    )
    def test_xss_input_handling(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
        payload: str,
    ) -> None:

        response = client.get(
            ASSETS,
            headers=admin_headers,
            params={"search": payload},
        )

        assert response.status_code in {200, 400, 422}

        assert "<script>" not in response.text

    def test_path_traversal_rejected(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
    ) -> None:

        response = client.get(
            f"{REPORTS}/download",
            headers=admin_headers,
            params={"filename": "../../etc/passwd"},
        )

        assert response.status_code in {400, 403, 404, 422}


# --------------------------------------------------
# Input Validation
# --------------------------------------------------

class TestInputValidation:

    def test_invalid_json(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
    ) -> None:

        response = client.post(
            ASSETS,
            headers={
                **admin_headers,
                "Content-Type": "application/json",
            },
            content="{invalid-json",
        )

        assert response.status_code in {400, 422}

    def test_invalid_telemetry_type(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
        telemetry_payload: dict[str, Any],
    ) -> None:

        payload = {
            **telemetry_payload,
            "temperature": "not-a-number",
        }

        response = client.post(
            TELEMETRY,
            headers=operator_headers,
            json=payload,
        )

        assert response.status_code == 422

    def test_negative_energy_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
        telemetry_payload: dict[str, Any],
    ) -> None:

        payload = {
            **telemetry_payload,
            "energy": -100.0,
        }

        response = client.post(
            TELEMETRY,
            headers=operator_headers,
            json=payload,
        )

        assert response.status_code == 422

    def test_mass_assignment_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
        asset_payload: dict[str, Any],
    ) -> None:

        payload = {
            **asset_payload,
            "is_admin": True,
            "role": "super_admin",
            "tenant_id": "OTHER-TENANT",
        }

        response = client.post(
            ASSETS,
            headers=operator_headers,
            json=payload,
        )

        assert response.status_code in {400, 403, 422}


# --------------------------------------------------
# Sensitive Information Leakage
# --------------------------------------------------

class TestInformationLeakage:

    def test_error_does_not_expose_traceback(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            f"{ASSETS}/nonexistent"
        )

        body = response.text.lower()

        assert "traceback (most recent call last)" not in body
        assert "site-packages" not in body
        assert "sqlalchemy.exc" not in body

    def test_auth_failure_does_not_expose_secret(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
            headers={
                "Authorization": "Bearer invalid"
            },
        )

        body = response.text.lower()

        assert "secret_key" not in body
        assert "jwt_secret" not in body
        assert "database_url" not in body

    def test_security_headers(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
        )

        assert (
            response.headers.get("x-content-type-options")
            == "nosniff"
        )

        assert response.headers.get(
            "x-frame-options"
        ) in {"DENY", "SAMEORIGIN"}


# --------------------------------------------------
# Industrial Safety API Protection
# --------------------------------------------------

class TestIndustrialSafetySecurity:

    def test_unauthorized_emergency_action_denied(
        self,
        client: TestClient,
    ) -> None:

        response = client.post(
            f"{EMERGENCY}/escalate",
            json={
                "incident_id": "TEST-INCIDENT-001",
                "escalation_level": "critical",
            },
        )

        assert response.status_code in {401, 403}

    def test_viewer_cannot_modify_emergency_status(
        self,
        client: TestClient,
        viewer_headers: dict[str, str],
    ) -> None:

        response = client.post(
            f"{EMERGENCY}/resolve",
            headers=viewer_headers,
            json={
                "incident_id": "TEST-INCIDENT-001",
            },
        )

        assert response.status_code == 403

    def test_unauthorized_safety_rule_change_denied(
        self,
        client: TestClient,
    ) -> None:

        response = client.put(
            f"{SAFETY}/rules",
            json={
                "rule_id": "FIRE-001",
                "enabled": False,
            },
        )

        assert response.status_code in {401, 403}

    def test_viewer_cannot_disable_fire_detection(
        self,
        client: TestClient,
        viewer_headers: dict[str, str],
    ) -> None:

        response = client.put(
            f"{SAFETY}/rules",
            headers=viewer_headers,
            json={
                "rule_id": "FIRE-DETECTION",
                "enabled": False,
            },
        )

        assert response.status_code == 403


# --------------------------------------------------
# Rate Limiting
# --------------------------------------------------

class TestRateLimiting:

    @pytest.mark.skip(
        reason=(
            "Requires isolated rate-limit storage, "
            "a documented limit, and deterministic "
            "test configuration."
        )
    )
    def test_login_rate_limiting(
        self,
        client: TestClient,
    ) -> None:

        responses = []

        for _ in range(20):
            response = client.post(
                f"{AUTH}/login",
                json={
                    "username": "invalid-user",
                    "password": "invalid-password",
                },
            )
            responses.append(response.status_code)

        assert 429 in responses


# --------------------------------------------------
# HTTP Method Security
# --------------------------------------------------

class TestHTTPMethodSecurity:

    def test_delete_not_allowed_on_telemetry_collection(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
    ) -> None:

        response = client.delete(
            TELEMETRY,
            headers=admin_headers,
        )

        assert response.status_code == 405

    def test_trace_method_disabled(
        self,
        client: TestClient,
    ) -> None:

        response = client.request(
            "TRACE",
            ASSETS,
        )

        assert response.status_code in {405, 501}


# --------------------------------------------------
# General API Reliability
# --------------------------------------------------

class TestAPISecurityReliability:

    def test_unknown_endpoint_returns_404(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            f"{API}/unknown-{uuid4().hex}"
        )

        assert response.status_code == 404

    def test_no_server_error_for_invalid_token(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            ASSETS,
            headers={
                "Authorization": "Bearer invalid-token"
            },
        )

        assert response.status_code in {401, 403}

    def test_health_endpoint_available(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            f"{API}/health"
        )

        assert response.status_code == 200
