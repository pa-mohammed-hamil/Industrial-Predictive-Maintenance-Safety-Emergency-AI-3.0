
"""
tests/security/test_file_upload_security.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

File Upload Security Tests

Run:
    pytest tests/security/test_file_upload_security.py -v

Proposed API:
    POST /api/v1/images/upload
    POST /api/v1/videos/upload
    POST /api/v1/inspections/upload
    POST /api/v1/assets/upload

Security coverage:
    - Authentication and authorization
    - File extension allowlists
    - MIME/content validation
    - File signature verification
    - Path traversal protection
    - Filename sanitization
    - Oversized upload rejection
    - Empty files
    - Malformed files
    - Suspicious document content
    - Cross-tenant upload isolation
    - File metadata safety

IMPORTANT:
Adapt the routes, schemas and authorization fixtures
to your implemented application.

All payloads are harmless synthetic test fixtures.
No live malware samples are used.
"""

from __future__ import annotations

import io
import os
import struct
import zlib
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


API = "/api/v1"

IMAGE_UPLOAD = f"{API}/images/upload"
VIDEO_UPLOAD = f"{API}/videos/upload"
INSPECTION_UPLOAD = f"{API}/inspections/upload"
ASSET_UPLOAD = f"{API}/assets/upload"

MAX_UPLOAD_BYTES = int(
    os.getenv("TEST_MAX_UPLOAD_BYTES", "10485760")
)


# --------------------------------------------------
# Safe synthetic file generators
# --------------------------------------------------

def make_png() -> bytes:
    """Create a valid minimal 1x1 PNG."""

    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(
                ">I",
                zlib.crc32(body) & 0xFFFFFFFF,
            )
        )

    ihdr = struct.pack(
        ">IIBBBBB",
        1, 1, 8, 2, 0, 0, 0,
    )

    raw = b"\x00\xff\x00\x00"
    compressed = zlib.compress(raw)

    return (
        signature
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def make_pdf() -> bytes:
    """Create a minimal synthetic PDF."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 200 200] >>"
        ),
    ]

    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(
            f"{index} 0 obj\n".encode()
            + obj
            + b"\nendobj\n"
        )

    xref_offset = len(result)

    result.extend(
        f"xref\n0 {len(objects) + 1}\n".encode()
    )
    result.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        result.extend(
            f"{offset:010d} 00000 n \n".encode()
        )

    result.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} "
            "/Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode()
    )

    return bytes(result)


def make_csv() -> bytes:
    return (
        b"timestamp,asset_id,temperature,vibration\n"
        b"2026-01-01T10:00:00Z,PUMP-001,45.2,2.5\n"
    )


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
    Replace with a real test JWT issued by your
    isolated authentication fixture.
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


def upload_file(
    client: TestClient,
    endpoint: str,
    filename: str,
    content: bytes,
    content_type: str,
    headers: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
):
    return client.post(
        endpoint,
        headers=headers,
        data=data,
        files={
            "file": (
                filename,
                io.BytesIO(content),
                content_type,
            )
        },
    )


# --------------------------------------------------
# Valid file uploads
# --------------------------------------------------

class TestValidUploads:

    def test_valid_png_upload(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "pump_inspection.png",
            make_png(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {200, 201}

    def test_valid_pdf_upload(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            INSPECTION_UPLOAD,
            "inspection_report.pdf",
            make_pdf(),
            "application/pdf",
            operator_headers,
        )

        assert response.status_code in {200, 201}

    def test_valid_csv_upload(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            ASSET_UPLOAD,
            "telemetry.csv",
            make_csv(),
            "text/csv",
            operator_headers,
        )

        assert response.status_code in {200, 201}


# --------------------------------------------------
# Authentication
# --------------------------------------------------

class TestUploadAuthentication:

    @pytest.mark.parametrize(
        "endpoint",
        [
            IMAGE_UPLOAD,
            VIDEO_UPLOAD,
            INSPECTION_UPLOAD,
            ASSET_UPLOAD,
        ],
    )
    def test_unauthenticated_upload_rejected(
        self,
        client: TestClient,
        endpoint: str,
    ) -> None:

        response = upload_file(
            client,
            endpoint,
            "test.png",
            make_png(),
            "image/png",
        )

        assert response.status_code in {401, 403}

    def test_invalid_token_rejected(
        self,
        client: TestClient,
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "test.png",
            make_png(),
            "image/png",
            {
                "Authorization": "Bearer invalid.token"
            },
        )

        assert response.status_code == 401


# --------------------------------------------------
# Role-based access
# --------------------------------------------------

class TestUploadAuthorization:

    def test_viewer_cannot_upload(
        self,
        client: TestClient,
        viewer_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "inspection.png",
            make_png(),
            "image/png",
            viewer_headers,
        )

        assert response.status_code == 403

    def test_operator_can_upload(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "inspection.png",
            make_png(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {200, 201}


# --------------------------------------------------
# Extension validation
# --------------------------------------------------

class TestExtensionValidation:

    @pytest.mark.parametrize(
        "filename",
        [
            "payload.exe",
            "script.sh",
            "program.bat",
            "command.cmd",
            "installer.msi",
            "script.ps1",
            "malicious.php",
            "code.py",
        ],
    )
    def test_executable_extensions_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
        filename: str,
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            filename,
            b"synthetic test content",
            "application/octet-stream",
            operator_headers,
        )

        assert response.status_code in {400, 415, 422}

    @pytest.mark.parametrize(
        "filename",
        [
            "inspection.png.exe",
            "inspection.jpg.php",
            "inspection.pdf.sh",
            "inspection.png.bat",
        ],
    )
    def test_double_extensions_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
        filename: str,
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            filename,
            make_png(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {400, 415, 422}


# --------------------------------------------------
# MIME and signature validation
# --------------------------------------------------

class TestContentValidation:

    def test_fake_png_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "fake.png",
            b"This is not a PNG image",
            "image/png",
            operator_headers,
        )

        assert response.status_code in {400, 415, 422}

    def test_mime_mismatch_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "inspection.png",
            make_png(),
            "application/pdf",
            operator_headers,
        )

        assert response.status_code in {400, 415, 422}

    def test_pdf_disguised_as_image(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "fake_image.png",
            make_pdf(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {400, 415, 422}

    def test_truncated_png_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "broken.png",
            b"\x89PNG\r\n\x1a\n",
            "image/png",
            operator_headers,
        )

        assert response.status_code in {400, 415, 422}


# --------------------------------------------------
# Path traversal protection
# --------------------------------------------------

class TestPathTraversal:

    @pytest.mark.parametrize(
        "filename",
        [
            "../inspection.png",
            "../../inspection.png",
            r"..\inspection.png",
            r"..\..\inspection.png",
            "/tmp/inspection.png",
        ],
    )
    def test_unsafe_filename_rejected_or_sanitized(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
        filename: str,
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            filename,
            make_png(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {
            200, 201, 400, 415, 422
        }

        if response.status_code in {200, 201}:
            result = response.json()

            stored_name = result.get(
                "stored_filename",
                result.get("filename", ""),
            )

            assert "/" not in stored_name
            assert "\\" not in stored_name
            assert ".." not in stored_name


# --------------------------------------------------
# Upload size limits
# --------------------------------------------------

class TestUploadSize:

    def test_empty_file_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "empty.png",
            b"",
            "image/png",
            operator_headers,
        )

        assert response.status_code in {400, 415, 422}

    def test_oversized_file_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        oversized = (
            make_png()
            + b"\x00" * (MAX_UPLOAD_BYTES + 1)
        )

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "oversized.png",
            oversized,
            "image/png",
            operator_headers,
        )

        assert response.status_code in {
            400, 413, 422
        }


# --------------------------------------------------
# Document safety
# --------------------------------------------------

class TestDocumentSafety:

    def test_synthetic_pdf_script_marker(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:
        """
        A harmless PDF-like document containing
        a JavaScript action marker.

        A strict document ingestion policy should
        reject active-content documents or sanitize
        them before further processing.
        """

        content = (
            b"%PDF-1.4\n"
            b"1 0 obj\n"
            b"<< /Type /Catalog /OpenAction "
            b"<< /S /JavaScript "
            b"/JS (synthetic_test_marker) >> >>\n"
            b"endobj\n"
            b"%%EOF\n"
        )

        response = upload_file(
            client,
            INSPECTION_UPLOAD,
            "active_content.pdf",
            content,
            "application/pdf",
            operator_headers,
        )

        assert response.status_code in {
            400, 415, 422
        }

    def test_csv_formula_injection_policy(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:
        """
        CSV formula injection becomes relevant when
        untrusted CSV cells are opened in spreadsheet
        software. This test assumes rejection rather
        than sanitization or export-time escaping.
        """

        content = (
            b"asset_id,description\n"
            b"PUMP-001,=1+1\n"
        )

        response = upload_file(
            client,
            ASSET_UPLOAD,
            "formula.csv",
            content,
            "text/csv",
            operator_headers,
        )

        assert response.status_code in {
            400, 415, 422
        }


# --------------------------------------------------
# Tenant isolation
# --------------------------------------------------

class TestUploadTenantIsolation:

    def test_cross_tenant_upload_rejected(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:
        """
        Requires a seeded asset belonging to a
        different tenant than the test operator.
        """

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "inspection.png",
            make_png(),
            "image/png",
            operator_headers,
            data={
                "asset_id": "OTHER-TENANT-ASSET"
            },
        )

        assert response.status_code in {403, 404}

    def test_client_cannot_override_tenant(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "inspection.png",
            make_png(),
            "image/png",
            operator_headers,
            data={
                "tenant_id": "OTHER-TENANT"
            },
        )

        assert response.status_code in {
            400, 403, 422
        }


# --------------------------------------------------
# File metadata protection
# --------------------------------------------------

class TestUploadMetadata:

    def test_uploaded_file_has_identifier(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            f"inspection-{uuid4().hex}.png",
            make_png(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {200, 201}

        result = response.json()

        assert result.get("file_id") or result.get("id")

    def test_upload_response_hides_storage_paths(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "inspection.png",
            make_png(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {200, 201}

        result = response.json()

        assert "absolute_path" not in result
        assert "storage_credentials" not in result
        assert "secret_key" not in result

    def test_uploaded_filename_is_not_trusted(
        self,
        client: TestClient,
        operator_headers: dict[str, str],
    ) -> None:

        response = upload_file(
            client,
            IMAGE_UPLOAD,
            "inspection<script>.png",
            make_png(),
            "image/png",
            operator_headers,
        )

        assert response.status_code in {
            200, 201, 400, 415, 422
        }

        if response.status_code in {200, 201}:
            result = response.json()

            stored_name = result.get(
                "stored_filename",
                result.get("filename", ""),
            )

            assert "<" not in stored_name
            assert ">" not in stored_name
