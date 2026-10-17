
"""
tests/security/test_auth.py

Authentication security tests for the Industrial AI platform.

Run:
    pytest tests/security/test_auth.py -v

Proposed authentication API contract:
    POST /api/v1/auth/login
    POST /api/v1/auth/refresh
    POST /api/v1/auth/logout
    GET  /api/v1/auth/me
    GET  /api/v1/assets
    GET  /api/v1/incidents

IMPORTANT:
The project structure does not specify authentication
schemas or endpoint behavior. These tests assume:

- Login accepts username and password as JSON.
- Login returns access_token, refresh_token, token_type.
- Refresh accepts refresh_token as JSON.
- Logout revokes the supplied refresh token.
- Protected endpoints require Bearer authentication.
- JWT access tokens use HS256 in the test environment.

Use a disposable test database and a test-only signing key.
Do not use real credentials or production services.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app


API = "/api/v1"
LOGIN = f"{API}/auth/login"
REFRESH = f"{API}/auth/refresh"
LOGOUT = f"{API}/auth/logout"
ME = f"{API}/auth/me"
ASSETS = f"{API}/assets"
INCIDENTS = f"{API}/incidents"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def credentials() -> dict[str, str]:
    """
    Seed this account in the isolated test database.

    Environment variables:
        TEST_AUTH_USERNAME
        TEST_AUTH_PASSWORD
    """
    return {
        "username": os.getenv(
            "TEST_AUTH_USERNAME",
            "test_operator",
        ),
        "password": os.getenv(
            "TEST_AUTH_PASSWORD",
            "TestPassword123!",
        ),
    }


@pytest.fixture
def valid_tokens(
    client: TestClient,
    credentials: dict[str, str],
) -> dict[str, Any]:
    response = client.post(LOGIN, json=credentials)

    assert response.status_code == 200, response.text

    data = response.json()

    assert isinstance(data, dict)
    assert isinstance(data.get("access_token"), str)
    assert isinstance(data.get("refresh_token"), str)

    return data


@pytest.fixture
def auth_headers(
    valid_tokens: dict[str, Any],
) -> dict[str, str]:
    return {
        "Authorization": (
            f"Bearer {valid_tokens['access_token']}"
        )
    }


def decode_jwt_payload(token: str) -> dict[str, Any]:
    """
    Decode JWT payload without verifying its signature.

    ONLY use this helper to inspect token structure.
    Never use it to authenticate or authorize a user.
    """
    parts = token.split(".")

    assert len(parts) == 3, "Expected a three-part JWT"

    payload = parts[1]
    padding = "=" * (-len(payload) % 4)

    decoded = base64.urlsafe_b64decode(
        payload + padding
    )

    result = json.loads(decoded)

    assert isinstance(result, dict)
    return result


def bearer(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}"
    }


class TestLogin:

    def test_valid_login(
        self,
        client: TestClient,
        credentials: dict[str, str],
    ) -> None:
        response = client.post(
            LOGIN,
            json=credentials,
        )

        assert response.status_code == 200

        data = response.json()

        assert data["access_token"]
        assert data["refresh_token"]
        assert data["token_type"].lower() == "bearer"

    def test_invalid_password(
        self,
        client: TestClient,
        credentials: dict[str, str],
    ) -> None:
        response = client.post(
            LOGIN,
            json={
                **credentials,
                "password": "IncorrectPassword999!",
            },
        )

        assert response.status_code in {400, 401}

        assert "access_token" not in response.json()

    def test_unknown_user(
        self,
        client: TestClient,
    ) -> None:
        response = client.post(
            LOGIN,
            json={
                "username": "unknown_test_account",
                "password": "InvalidPassword123!",
            },
        )

        assert response.status_code in {400, 401}

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"username": "test_operator"},
            {"password": "TestPassword123!"},
            {"username": "", "password": ""},
            {"username": None, "password": None},
        ],
    )
    def test_missing_or_invalid_credentials(
        self,
        client: TestClient,
        payload: dict[str, Any],
    ) -> None:
        response = client.post(
            LOGIN,
            json=payload,
        )

        assert response.status_code in {400, 401, 422}

    def test_login_response_does_not_expose_password(
        self,
        client: TestClient,
        credentials: dict[str, str],
    ) -> None:
        response = client.post(
            LOGIN,
            json=credentials,
        )

        assert response.status_code == 200

        body = response.text.lower()

        assert credentials["password"].lower() not in body
        assert "password_hash" not in body
        assert "hashed_password" not in body


class TestAccessTokens:

    def test_access_token_is_jwt(
        self,
        valid_tokens: dict[str, Any],
    ) -> None:
        token = valid_tokens["access_token"]

        payload = decode_jwt_payload(token)

        assert "sub" in payload
        assert "exp" in payload

    def test_access_token_expiry_is_future(
        self,
        valid_tokens: dict[str, Any],
    ) -> None:
        payload = decode_jwt_payload(
            valid_tokens["access_token"]
        )

        now = datetime.now(timezone.utc).timestamp()

        assert payload["exp"] > now

    def test_access_token_has_subject(
        self,
        valid_tokens: dict[str, Any],
    ) -> None:
        payload = decode_jwt_payload(
            valid_tokens["access_token"]
        )

        assert isinstance(payload["sub"], str)
        assert payload["sub"]

    def test_token_has_valid_expiry_duration(
        self,
        valid_tokens: dict[str, Any],
    ) -> None:
        payload = decode_jwt_payload(
            valid_tokens["access_token"]
        )

        issued = payload.get("iat")
        expires = payload["exp"]

        assert issued is not None
        assert expires > issued

        # Example test policy: access tokens last
        # no longer than 60 minutes.
        assert expires - issued <= 3600


class TestProtectedEndpoints:

    @pytest.mark.parametrize(
        "endpoint",
        [
            ME,
            ASSETS,
            INCIDENTS,
        ],
    )
    def test_missing_authentication(
        self,
        client: TestClient,
        endpoint: str,
    ) -> None:
        response = client.get(endpoint)

        assert response.status_code in {401, 403}

    @pytest.mark.parametrize(
        "endpoint",
        [
            ME,
            ASSETS,
            INCIDENTS,
        ],
    )
    def test_valid_authentication(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        endpoint: str,
    ) -> None:
        response = client.get(
            endpoint,
            headers=auth_headers,
        )

        assert response.status_code == 200

    def test_invalid_token(
        self,
        client: TestClient,
    ) -> None:
        response = client.get(
            ME,
            headers=bearer("invalid.token.value"),
        )

        assert response.status_code == 401

    def test_empty_bearer_token(
        self,
        client: TestClient,
    ) -> None:
        response = client.get(
            ME,
            headers=bearer(""),
        )

        assert response.status_code in {401, 403}

    def test_malformed_auth_header(
        self,
        client: TestClient,
    ) -> None:
        response = client.get(
            ME,
            headers={
                "Authorization": "Basic invalidcredentials"
            },
        )

        assert response.status_code in {401, 403}


class TestTokenTampering:

    def test_modified_signature_rejected(
        self,
        client: TestClient,
        valid_tokens: dict[str, Any],
    ) -> None:
        token = valid_tokens["access_token"]
        parts = token.split(".")

        assert len(parts) == 3

        signature = parts[2]
        assert signature

        replacement = (
            "A" if signature[-1] != "A" else "B"
        )

        tampered_signature = (
            signature[:-1] + replacement
        )

        tampered = ".".join(
            [parts[0], parts[1], tampered_signature]
        )

        response = client.get(
            ME,
            headers=bearer(tampered),
        )

        assert response.status_code == 401

    def test_modified_payload_rejected(
        self,
        client: TestClient,
        valid_tokens: dict[str, Any],
    ) -> None:
        token = valid_tokens["access_token"]
        parts = token.split(".")

        payload = decode_jwt_payload(token)

        payload["sub"] = "unauthorized_admin"

        encoded = base64.urlsafe_b64encode(
            json.dumps(
                payload,
                separators=(",", ":"),
            ).encode()
        ).rstrip(b"=").decode()

        tampered = ".".join(
            [parts[0], encoded, parts[2]]
        )

        response = client.get(
            ME,
            headers=bearer(tampered),
        )

        assert response.status_code == 401


class TestExpiredTokens:

    def test_expired_token_rejected(
        self,
        client: TestClient,
    ) -> None:
        """
        Requires a test-only JWT signing key.

        This test constructs a correctly signed
        but expired token to verify expiry checks.
        """
        jwt = pytest.importorskip("jwt")

        secret = os.getenv("TEST_JWT_SECRET")

        if not secret:
            pytest.skip(
                "Set TEST_JWT_SECRET for isolated tests"
            )

        now = datetime.now(timezone.utc)

        token = jwt.encode(
            {
                "sub": "test_operator",
                "iat": now - timedelta(hours=2),
                "exp": now - timedelta(hours=1),
                "type": "access",
            },
            secret,
            algorithm="HS256",
        )

        response = client.get(
            ME,
            headers=bearer(token),
        )

        assert response.status_code == 401

    def test_unsigned_jwt_rejected(
        self,
        client: TestClient,
    ) -> None:
        jwt = pytest.importorskip("jwt")

        now = datetime.now(timezone.utc)

        token = jwt.encode(
            {
                "sub": "test_operator",
                "iat": now,
                "exp": now + timedelta(minutes=30),
            },
            key="",
            algorithm="none",
        )

        response = client.get(
            ME,
            headers=bearer(token),
        )

        assert response.status_code == 401


class TestRefreshTokens:

    def test_refresh_access_token(
        self,
        client: TestClient,
        valid_tokens: dict[str, Any],
    ) -> None:
        response = client.post(
            REFRESH,
            json={
                "refresh_token": (
                    valid_tokens["refresh_token"]
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["access_token"]
        assert data["token_type"].lower() == "bearer"

    def test_invalid_refresh_token(
        self,
        client: TestClient,
    ) -> None:
        response = client.post(
            REFRESH,
            json={
                "refresh_token": "invalid-refresh-token"
            },
        )

        assert response.status_code == 401

    def test_access_token_cannot_be_used_as_refresh(
        self,
        client: TestClient,
        valid_tokens: dict[str, Any],
    ) -> None:
        response = client.post(
            REFRESH,
            json={
                "refresh_token": (
                    valid_tokens["access_token"]
                )
            },
        )

        assert response.status_code == 401


class TestLogout:

    def test_logout_success(
        self,
        client: TestClient,
        valid_tokens: dict[str, Any],
    ) -> None:
        response = client.post(
            LOGOUT,
            json={
                "refresh_token": (
                    valid_tokens["refresh_token"]
                )
            },
            headers=bearer(
                valid_tokens["access_token"]
            ),
        )

        assert response.status_code in {200, 204}

    def test_revoked_refresh_token_rejected(
        self,
        client: TestClient,
        valid_tokens: dict[str, Any],
    ) -> None:
        refresh_token = valid_tokens["refresh_token"]

        logout = client.post(
            LOGOUT,
            json={"refresh_token": refresh_token},
            headers=bearer(
                valid_tokens["access_token"]
            ),
        )

        assert logout.status_code in {200, 204}

        response = client.post(
            REFRESH,
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == 401


class TestCurrentUser:

    def test_current_user_response(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        response = client.get(
            ME,
            headers=auth_headers,
        )

        assert response.status_code == 200

        data = response.json()

        assert "id" in data
        assert "username" in data

    def test_current_user_hides_sensitive_fields(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        response = client.get(
            ME,
            headers=auth_headers,
        )

        assert response.status_code == 200

        data = response.json()

        forbidden = {
            "password",
            "password_hash",
            "hashed_password",
            "secret_key",
            "refresh_token",
        }

        assert forbidden.isdisjoint(data.keys())


class TestAuthenticationErrorHandling:

    def test_auth_errors_do_not_expose_secrets(
        self,
        client: TestClient,
    ) -> None:
        response = client.post(
            LOGIN,
            json={
                "username": "unknown_user",
                "password": "wrong_password",
            },
        )

        body = response.text.lower()

        forbidden = [
            "secret_key",
            "jwt_secret",
            "database_url",
            "password_hash",
            "traceback (most recent call last)",
        ]

        for item in forbidden:
            assert item not in body

    def test_invalid_token_does_not_return_500(
        self,
        client: TestClient,
    ) -> None:
        response = client.get(
            ME,
            headers=bearer("invalid"),
        )

        assert response.status_code in {401, 403}
