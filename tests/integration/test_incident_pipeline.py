
"""
tests/integration/test_incident_pipeline.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI.

Integration tests for incident management and
emergency response workflows.

Run:
    pytest tests/integration/test_incident_pipeline.py -v

IMPORTANT:
The uploaded project specifies file names and modules,
not the implemented API contracts.

The routes, payloads, and response schemas below are
proposed contracts. Adapt them to your actual backend.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


API = "/api/v1"

INCIDENTS = f"{API}/incidents"
SAFETY = f"{API}/safety"
EMERGENCY = f"{API}/emergency"
NOTIFICATIONS = f"{API}/notifications"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def incident_payload() -> dict[str, Any]:
    return {
        "asset_id": "PUMP-001",
        "site_id": "PLANT-001",
        "zone_id": "ZONE-A",
        "incident_type": "overheating",
        "severity": "high",
        "description": "Abnormal motor temperature",
        "detected_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sensor_data": {
            "temperature": 105.0,
            "vibration": 7.5,
            "pressure": 4.2,
        },
        "source": "sensor",
    }


@pytest.fixture
def created_incident(
    client: TestClient,
    incident_payload: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(
        INCIDENTS,
        json=incident_payload,
    )

    assert response.status_code == 201, response.text

    result = response.json()
    assert "id" in result

    return result


def request_json(
    client: TestClient,
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:

    response = client.request(
        method,
        endpoint,
        json=payload,
    )

    assert response.status_code == expected_status, (
        f"{method} {endpoint}\n"
        f"Expected: {expected_status}\n"
        f"Actual: {response.status_code}\n"
        f"Response: {response.text}"
    )

    result = response.json()

    assert isinstance(result, dict)
    return result


class TestIncidentCreation:

    def test_create_incident(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "POST",
            INCIDENTS,
            incident_payload,
            expected_status=201,
        )

        assert result["id"]
        assert result["asset_id"] == "PUMP-001"
        assert result["incident_type"] == "overheating"
        assert result["severity"] == "high"

    def test_unique_incident_ids(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
    ) -> None:

        first = request_json(
            client,
            "POST",
            INCIDENTS,
            incident_payload,
            201,
        )

        second = request_json(
            client,
            "POST",
            INCIDENTS,
            incident_payload,
            201,
        )

        assert first["id"] != second["id"]

    def test_incident_timestamp(
        self,
        created_incident: dict[str, Any],
    ) -> None:

        timestamp = created_incident["detected_at"]

        parsed = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

        assert parsed.tzinfo is not None

    def test_default_status(
        self,
        created_incident: dict[str, Any],
    ) -> None:

        assert created_incident["status"] in {
            "open",
            "detected",
            "active",
        }


class TestIncidentRetrieval:

    def test_get_incident_by_id(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        incident_id = created_incident["id"]

        result = request_json(
            client,
            "GET",
            f"{INCIDENTS}/{incident_id}",
        )

        assert result["id"] == incident_id

    def test_incident_not_found(
        self,
        client: TestClient,
    ) -> None:

        unknown_id = str(uuid4())

        response = client.get(
            f"{INCIDENTS}/{unknown_id}"
        )

        assert response.status_code == 404

    def test_filter_incidents_by_asset(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        response = client.get(
            INCIDENTS,
            params={"asset_id": "PUMP-001"},
        )

        assert response.status_code == 200

        result = response.json()

        assert isinstance(result, list)

        assert any(
            item["id"] == created_incident["id"]
            for item in result
        )

        assert all(
            item["asset_id"] == "PUMP-001"
            for item in result
        )


class TestIncidentSeverity:

    @pytest.mark.parametrize(
        "severity",
        ["low", "medium", "high", "critical"],
    )
    def test_valid_severity_levels(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
        severity: str,
    ) -> None:

        payload = {
            **incident_payload,
            "severity": severity,
        }

        result = request_json(
            client,
            "POST",
            INCIDENTS,
            payload,
            201,
        )

        assert result["severity"] == severity

    def test_invalid_severity_rejected(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
    ) -> None:

        payload = {
            **incident_payload,
            "severity": "extremely_super_high",
        }

        response = client.post(
            INCIDENTS,
            json=payload,
        )

        assert response.status_code == 422


class TestIncidentDetection:

    def test_overheating_detection(
        self,
        client: TestClient,
    ) -> None:

        payload = {
            "asset_id": "PUMP-001",
            "temperature": 110.0,
            "threshold": 85.0,
        }

        result = request_json(
            client,
            "POST",
            f"{SAFETY}/detect/overheating",
            payload,
        )

        assert result["detected"] is True

    def test_normal_temperature(
        self,
        client: TestClient,
    ) -> None:

        payload = {
            "asset_id": "PUMP-001",
            "temperature": 45.0,
            "threshold": 85.0,
        }

        result = request_json(
            client,
            "POST",
            f"{SAFETY}/detect/overheating",
            payload,
        )

        assert result["detected"] is False

    def test_fire_detection(
        self,
        client: TestClient,
    ) -> None:

        payload = {
            "asset_id": "PUMP-001",
            "fire_confidence": 0.96,
            "threshold": 0.80,
        }

        result = request_json(
            client,
            "POST",
            f"{SAFETY}/detect/fire",
            payload,
        )

        assert result["detected"] is True

    def test_smoke_detection(
        self,
        client: TestClient,
    ) -> None:

        payload = {
            "asset_id": "PUMP-001",
            "smoke_confidence": 0.91,
            "threshold": 0.80,
        }

        result = request_json(
            client,
            "POST",
            f"{SAFETY}/detect/smoke",
            payload,
        )

        assert result["detected"] is True


class TestIncidentCorrelation:

    def test_multi_signal_correlation(
        self,
        client: TestClient,
    ) -> None:

        payload = {
            "asset_id": "PUMP-001",
            "signals": {
                "temperature": 115.0,
                "vibration": 12.0,
                "smoke_detected": True,
                "fire_detected": False,
            },
        }

        result = request_json(
            client,
            "POST",
            f"{INCIDENTS}/correlate",
            payload,
        )

        assert "risk_score" in result
        assert 0 <= result["risk_score"] <= 1

        assert "severity" in result

    def test_empty_signals_rejected(
        self,
        client: TestClient,
    ) -> None:

        response = client.post(
            f"{INCIDENTS}/correlate",
            json={
                "asset_id": "PUMP-001",
                "signals": {},
            },
        )

        assert response.status_code == 422


class TestIncidentLifecycle:

    def test_acknowledge_incident(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        incident_id = created_incident["id"]

        result = request_json(
            client,
            "PATCH",
            f"{INCIDENTS}/{incident_id}/acknowledge",
            {
                "acknowledged_by": "operator-001",
            },
        )

        assert result["status"] == "acknowledged"

    def test_resolve_incident(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        incident_id = created_incident["id"]

        request_json(
            client,
            "PATCH",
            f"{INCIDENTS}/{incident_id}/acknowledge",
            {"acknowledged_by": "operator-001"},
        )

        result = request_json(
            client,
            "PATCH",
            f"{INCIDENTS}/{incident_id}/resolve",
            {
                "resolved_by": "engineer-001",
                "resolution_notes": (
                    "Cooling system repaired"
                ),
            },
        )

        assert result["status"] == "resolved"

    def test_closed_incident_cannot_be_reopened(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        incident_id = created_incident["id"]

        request_json(
            client,
            "PATCH",
            f"{INCIDENTS}/{incident_id}/acknowledge",
            {"acknowledged_by": "operator-001"},
        )

        request_json(
            client,
            "PATCH",
            f"{INCIDENTS}/{incident_id}/resolve",
            {
                "resolved_by": "engineer-001",
                "resolution_notes": "Issue fixed",
            },
        )

        request_json(
            client,
            "PATCH",
            f"{INCIDENTS}/{incident_id}/close",
            {"closed_by": "supervisor-001"},
        )

        response = client.patch(
            f"{INCIDENTS}/{incident_id}/acknowledge",
            json={"acknowledged_by": "operator-002"},
        )

        assert response.status_code == 409


class TestEmergencyResponse:

    def test_critical_incident_escalation(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
    ) -> None:

        payload = {
            **incident_payload,
            "severity": "critical",
            "incident_type": "fire",
        }

        incident = request_json(
            client,
            "POST",
            INCIDENTS,
            payload,
            201,
        )

        result = request_json(
            client,
            "POST",
            f"{EMERGENCY}/escalate",
            {
                "incident_id": incident["id"],
                "escalation_level": "critical",
            },
        )

        assert result["escalated"] is True
        assert result["incident_id"] == incident["id"]

    def test_emergency_response_plan(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "POST",
            f"{EMERGENCY}/response-plan",
            {
                "incident_id": created_incident["id"],
            },
        )

        assert "actions" in result
        assert isinstance(result["actions"], list)
        assert len(result["actions"]) > 0


class TestIncidentNotifications:

    def test_critical_incident_notification(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
    ) -> None:

        payload = {
            **incident_payload,
            "severity": "critical",
        }

        incident = request_json(
            client,
            "POST",
            INCIDENTS,
            payload,
            201,
        )

        result = request_json(
            client,
            "GET",
            f"{NOTIFICATIONS}/incident/{incident['id']}",
        )

        assert "notifications" in result
        assert isinstance(result["notifications"], list)


class TestIncidentValidation:

    @pytest.mark.parametrize(
        "missing_field",
        [
            "asset_id",
            "site_id",
            "incident_type",
            "severity",
        ],
    )
    def test_missing_required_fields(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
        missing_field: str,
    ) -> None:

        payload = deepcopy(incident_payload)
        payload.pop(missing_field)

        response = client.post(
            INCIDENTS,
            json=payload,
        )

        assert response.status_code == 422

    def test_invalid_sensor_data(
        self,
        client: TestClient,
        incident_payload: dict[str, Any],
    ) -> None:

        payload = {
            **incident_payload,
            "sensor_data": "invalid-data",
        }

        response = client.post(
            INCIDENTS,
            json=payload,
        )

        assert response.status_code == 422


class TestIncidentReliability:

    def test_incident_persistence(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        incident_id = created_incident["id"]

        first = request_json(
            client,
            "GET",
            f"{INCIDENTS}/{incident_id}",
        )

        second = request_json(
            client,
            "GET",
            f"{INCIDENTS}/{incident_id}",
        )

        assert first["id"] == second["id"]
        assert first["incident_type"] == second["incident_type"]

    def test_incident_asset_consistency(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        assert created_incident["asset_id"] == "PUMP-001"
        assert created_incident["site_id"] == "PLANT-001"

    def test_incident_audit_history(
        self,
        client: TestClient,
        created_incident: dict[str, Any],
    ) -> None:

        incident_id = created_incident["id"]

        result = request_json(
            client,
            "GET",
            f"{INCIDENTS}/{incident_id}/audit",
        )

        assert "events" in result
        assert isinstance(result["events"], list)
        assert len(result["events"]) >= 1
