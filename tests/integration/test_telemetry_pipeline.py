
"""
tests/integration/test_telemetry_pipeline.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Telemetry pipeline integration tests.

Run:
    pytest tests/integration/test_telemetry_pipeline.py -v

Proposed API contract:
    POST /api/v1/telemetry
    GET  /api/v1/telemetry/{id}
    GET  /api/v1/telemetry/history
    POST /api/v1/telemetry/batch
    POST /api/v1/anomalies/detect
    POST /api/v1/predictions/health-score
    WS   /api/v1/ws/telemetry

Adapt routes and schemas to the actual implementation.

Use isolated test databases, mock message brokers and
simulated sensors. Never publish test readings to
production industrial equipment.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


API = "/api/v1"
TELEMETRY = f"{API}/telemetry"
ANOMALIES = f"{API}/anomalies"
PREDICTIONS = f"{API}/predictions"
WS_TELEMETRY = f"{API}/ws/telemetry"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def telemetry_payload() -> dict[str, Any]:
    return {
        "asset_id": "MACHINE-001",
        "sensor_id": "SENSOR-001",
        "site_id": "PLANT-001",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "temperature": 45.0,
        "vibration": 2.5,
        "pressure": 4.2,
        "current": 12.0,
        "voltage": 230.0,
        "energy": 5.5,
    }


@pytest.fixture
def ingested_telemetry(
    client: TestClient,
    telemetry_payload: dict[str, Any],
) -> dict[str, Any]:

    response = client.post(
        TELEMETRY,
        json=telemetry_payload,
    )

    assert response.status_code == 201, response.text

    result = response.json()
    assert "id" in result
    return result


def request_json(
    client: TestClient,
    method: str,
    endpoint: str,
    payload: Any = None,
    expected_status: int = 200,
    params: dict[str, Any] | None = None,
) -> Any:

    response = client.request(
        method,
        endpoint,
        json=payload,
        params=params,
    )

    assert response.status_code == expected_status, (
        f"{method} {endpoint}\n"
        f"Expected: {expected_status}\n"
        f"Actual: {response.status_code}\n"
        f"Response: {response.text}"
    )

    return response.json()


class TestTelemetryIngestion:

    def test_ingest_telemetry(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "POST",
            TELEMETRY,
            telemetry_payload,
            201,
        )

        assert result["id"]
        assert result["asset_id"] == "MACHINE-001"
        assert result["sensor_id"] == "SENSOR-001"

    def test_unique_telemetry_ids(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        first = request_json(
            client, "POST", TELEMETRY,
            telemetry_payload, 201,
        )

        second_payload = {
            **telemetry_payload,
            "timestamp": (
                datetime.now(timezone.utc)
                + timedelta(seconds=1)
            ).isoformat(),
        }

        second = request_json(
            client, "POST", TELEMETRY,
            second_payload, 201,
        )

        assert first["id"] != second["id"]

    def test_temperature_preserved(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        assert ingested_telemetry["temperature"] == pytest.approx(
            45.0
        )

    def test_vibration_preserved(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        assert ingested_telemetry["vibration"] == pytest.approx(
            2.5
        )

    def test_pressure_preserved(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        assert ingested_telemetry["pressure"] == pytest.approx(
            4.2
        )

    def test_current_preserved(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        assert ingested_telemetry["current"] == pytest.approx(
            12.0
        )

    def test_voltage_preserved(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        assert ingested_telemetry["voltage"] == pytest.approx(
            230.0
        )

    def test_energy_preserved(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        assert ingested_telemetry["energy"] == pytest.approx(
            5.5
        )


class TestTelemetryRetrieval:

    def test_get_telemetry_by_id(
        self,
        client: TestClient,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        telemetry_id = ingested_telemetry["id"]

        result = request_json(
            client,
            "GET",
            f"{TELEMETRY}/{telemetry_id}",
        )

        assert result["id"] == telemetry_id

    def test_unknown_telemetry_returns_404(
        self,
        client: TestClient,
    ) -> None:

        response = client.get(
            f"{TELEMETRY}/{uuid4()}"
        )

        assert response.status_code == 404

    def test_telemetry_history(
        self,
        client: TestClient,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "GET",
            f"{TELEMETRY}/history",
            params={"asset_id": "MACHINE-001"},
        )

        assert isinstance(result, list)

        assert any(
            item["id"] == ingested_telemetry["id"]
            for item in result
        )

    def test_history_asset_filter(
        self,
        client: TestClient,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "GET",
            f"{TELEMETRY}/history",
            params={"asset_id": "MACHINE-001"},
        )

        assert all(
            item["asset_id"] == "MACHINE-001"
            for item in result
        )


class TestBatchTelemetry:

    def test_batch_ingestion(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        base = datetime.now(timezone.utc)

        records = []

        for index in range(10):
            records.append({
                **telemetry_payload,
                "timestamp": (
                    base + timedelta(seconds=index)
                ).isoformat(),
                "temperature": 40.0 + index,
            })

        result = request_json(
            client,
            "POST",
            f"{TELEMETRY}/batch",
            {"records": records},
            201,
        )

        assert result["ingested_count"] == 10

    def test_empty_batch_rejected(
        self,
        client: TestClient,
    ) -> None:

        response = client.post(
            f"{TELEMETRY}/batch",
            json={"records": []},
        )

        assert response.status_code == 422


class TestTelemetryAnomalies:

    def test_normal_telemetry(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "POST",
            f"{ANOMALIES}/detect",
            telemetry_payload,
        )

        assert "is_anomaly" in result
        assert isinstance(result["is_anomaly"], bool)

    def test_high_temperature_anomaly(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        payload = {
            **telemetry_payload,
            "temperature": 150.0,
        }

        result = request_json(
            client,
            "POST",
            f"{ANOMALIES}/detect",
            payload,
        )

        assert result["is_anomaly"] is True

    def test_high_vibration_anomaly(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        payload = {
            **telemetry_payload,
            "vibration": 25.0,
        }

        result = request_json(
            client,
            "POST",
            f"{ANOMALIES}/detect",
            payload,
        )

        assert result["is_anomaly"] is True

    def test_anomaly_score_range(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "POST",
            f"{ANOMALIES}/detect",
            telemetry_payload,
        )

        assert 0 <= result["anomaly_score"] <= 1


class TestMachineHealth:

    def test_machine_health_score(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        result = request_json(
            client,
            "POST",
            f"{PREDICTIONS}/health-score",
            telemetry_payload,
        )

        assert "health_score" in result
        assert 0 <= result["health_score"] <= 100

    def test_health_decreases_under_stress(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        normal = request_json(
            client,
            "POST",
            f"{PREDICTIONS}/health-score",
            telemetry_payload,
        )

        stressed_payload = {
            **telemetry_payload,
            "temperature": 140.0,
            "vibration": 20.0,
            "current": 35.0,
        }

        stressed = request_json(
            client,
            "POST",
            f"{PREDICTIONS}/health-score",
            stressed_payload,
        )

        assert (
            stressed["health_score"]
            <= normal["health_score"]
        )


class TestTelemetryValidation:

    @pytest.mark.parametrize(
        "field",
        ["asset_id", "sensor_id", "timestamp"],
    )
    def test_missing_required_fields(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
        field: str,
    ) -> None:

        payload = deepcopy(telemetry_payload)
        payload.pop(field)

        response = client.post(
            TELEMETRY,
            json=payload,
        )

        assert response.status_code == 422

    def test_invalid_temperature_type(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        payload = {
            **telemetry_payload,
            "temperature": "invalid",
        }

        response = client.post(
            TELEMETRY,
            json=payload,
        )

        assert response.status_code == 422

    def test_invalid_timestamp(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        payload = {
            **telemetry_payload,
            "timestamp": "not-a-date",
        }

        response = client.post(
            TELEMETRY,
            json=payload,
        )

        assert response.status_code == 422

    def test_negative_energy_rejected(
        self,
        client: TestClient,
        telemetry_payload: dict[str, Any],
    ) -> None:

        payload = {
            **telemetry_payload,
            "energy": -5.0,
        }

        response = client.post(
            TELEMETRY,
            json=payload,
        )

        assert response.status_code == 422


class TestTelemetryReliability:

    def test_telemetry_persistence(
        self,
        client: TestClient,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        telemetry_id = ingested_telemetry["id"]

        first = request_json(
            client,
            "GET",
            f"{TELEMETRY}/{telemetry_id}",
        )

        second = request_json(
            client,
            "GET",
            f"{TELEMETRY}/{telemetry_id}",
        )

        assert first["id"] == second["id"]
        assert first["temperature"] == second["temperature"]

    def test_asset_identity_preserved(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        assert ingested_telemetry["asset_id"] == "MACHINE-001"
        assert ingested_telemetry["site_id"] == "PLANT-001"

    def test_timestamp_is_timezone_aware(
        self,
        ingested_telemetry: dict[str, Any],
    ) -> None:

        timestamp = datetime.fromisoformat(
            ingested_telemetry["timestamp"].replace(
                "Z", "+00:00"
            )
        )

        assert timestamp.tzinfo is not None


class TestTelemetryStreaming:

    @pytest.mark.timeout(5)
    def test_websocket_telemetry_stream(
        self,
        client: TestClient,
    ) -> None:
        """
        Requires a WebSocket endpoint that sends a
        welcome/ready event immediately upon connection.

        Do not run against an endpoint that waits for
        external telemetry before sending a message.
        """

        with client.websocket_connect(
            WS_TELEMETRY
        ) as websocket:

            websocket.send_json({
                "action": "subscribe",
                "asset_id": "MACHINE-001",
            })

            message = websocket.receive_json()

            assert isinstance(message, dict)
            assert message.get("type") in {
                "connected",
                "subscribed",
                "telemetry",
            }
