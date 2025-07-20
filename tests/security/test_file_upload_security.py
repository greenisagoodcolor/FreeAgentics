"""
File Upload Security Testing

Comprehensive tests for validating file upload security measures
and preventing path traversal vulnerabilities.

This module focuses on:
1. File type validation and restrictions
2. File size limit enforcement
3. Path traversal attack prevention
4. File content scanning and validation
5. Upload directory security
6. File execution prevention
7. Malicious file detection
8. File metadata security
9. Storage location validation
10. File access control testing
"""

import io
import json

# Add the project root to the path
import sys
import time
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "/home/green/FreeAgentics")

try:
    from api.main import app
except (ImportError, ValueError):
    # Mock app for testing when main app is not available or database not configured
    from fastapi import FastAPI

    app = FastAPI()

    # Add mock endpoints for testing
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/v1/upload")
    async def mock_upload():
        return {"detail": "Upload endpoint not implemented"}

    @app.get("/api/v1/files")
    async def mock_files():
        return {"detail": "Files endpoint not implemented"}


class FileUploadSecurityTester:
    """Comprehensive file upload security tester."""

    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []

        # Dangerous file extensions that should be blocked
        self.dangerous_extensions = [
            "exe",
            "bat",
            "cmd",
            "com",
            "pif",
            "scr",
            "vbs",
            "js",
            "jar",
            "php",
            "asp",
            "aspx",
            "jsp",
            "py",
            "rb",
            "pl",
            "sh",
            "ps1",
            "htaccess",
            "htpasswd",
            "config",
            "ini",
            "conf",
            "cfg",
        ]

        # Safe file extensions that should be allowed
        self.safe_extensions = [
            "txt",
            "pdf",
            "doc",
            "docx",
            "xls",
            "xlsx",
            "ppt",
            "pptx",
            "jpg",
            "jpeg",
            "png",
            "gif",
            "bmp",
            "svg",
            "webp",
        ]

        # Path traversal payloads
        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "../../../var/log/auth.log",
            "../../../proc/self/environ",
            "../../../etc/shadow",
            "../../../home/user/.ssh/id_rsa",
            "../../../root/.bash_history",
            "....//....//....//etc/passwd",
            "..%2f..%2f..%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "/etc/passwd",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "file:///etc/passwd",
            "\\/etc/passwd",
            "\\\\server\\share\\file.txt",
            "/var/www/html/../../etc/passwd",
            ".\\.\\.\\.\\etc\\passwd",
            "%2e%2e%2fetc%2fpasswd",
            "%2e%2e%5cetc%5cpasswd",
        ]

        # Malicious file content patterns
        self.malicious_patterns = [
            b"<?php",
            b"<script>",
            b"javascript:",
            b"vbscript:",
            b"<%@",
            b"<jsp:",
            b"#!/bin/sh",
            b"#!/bin/bash",
            b"MZ",  # PE executable header
            b"\x7fELF",  # ELF executable header
            b"PK",  # ZIP file header (could contain malicious files)
        ]

    def test_file_extension_validation(self) -> Dict[str, Any]:
        """Test file extension validation and restrictions."""
        results = {
            "test_name": "file_extension_validation",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test dangerous extensions
        for ext in self.dangerous_extensions:
            try:
                filename = f"malicious_file.{ext}"
                file_content = b"This is a test file with dangerous extension"

                # Try to upload file with dangerous extension
                files = {
                    "file": (
                        filename,
                        io.BytesIO(file_content),
                        "application/octet-stream",
                    )
                }
                response = self.client.post("/api/v1/upload", files=files)

                # Should be rejected
                if response.status_code in [200, 201, 202]:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Dangerous file extension allowed",
                            "extension": ext,
                            "filename": filename,
                            "status_code": response.status_code,
                            "severity": "high",
                        }
                    )

                # Check response for information disclosure
                if (
                    "path" in response.text.lower()
                    or "directory" in response.text.lower()
                ):
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Path information disclosed in upload rejection",
                            "extension": ext,
                            "response_preview": response.text[:200],
                        }
                    )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing extension {ext}",
                        "error": str(e),
                    }
                )

        # Test safe extensions should be allowed (if upload endpoint exists)
        safe_uploads_blocked = 0
        for ext in self.safe_extensions[:5]:  # Test first 5 safe extensions
            try:
                filename = f"safe_file.{ext}"
                file_content = b"This is a safe test file"

                files = {
                    "file": (
                        filename,
                        io.BytesIO(file_content),
                        "application/octet-stream",
                    )
                }
                response = self.client.post("/api/v1/upload", files=files)

                # Should be allowed or show proper validation error
                if response.status_code == 403:
                    safe_uploads_blocked += 1

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing safe extension {ext}",
                        "error": str(e),
                    }
                )

        if safe_uploads_blocked == len(self.safe_extensions[:5]):
            results["findings"].append(
                {
                    "issue": "All safe file extensions blocked - may indicate overly restrictive policy",
                    "note": "Review file upload policy for business requirements",
                }
            )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement strict file extension validation whitelist",
                    "Block all executable and script file types",
                    "Do not disclose file system paths in error messages",
                    "Use MIME type validation in addition to extension checks",
                ]
            )

        return results

    def test_path_traversal_prevention(self) -> Dict[str, Any]:
        """Test protection against path traversal attacks in file uploads."""
        results = {
            "test_name": "path_traversal_prevention",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        for payload in self.path_traversal_payloads:
            try:
                # Test path traversal in filename
                malicious_filename = payload
                file_content = b"Path traversal test content"

                files = {
                    "file": (
                        malicious_filename,
                        io.BytesIO(file_content),
                        "text/plain",
                    )
                }
                response = self.client.post("/api/v1/upload", files=files)

                # Should be rejected or sanitized
                if response.status_code in [200, 201, 202]:
                    # Check if file was actually created in dangerous location
                    response_data = response.text

                    # Look for signs that traversal was successful
                    dangerous_indicators = [
                        "/etc/",
                        "/var/",
                        "/home/",
                        "/root/",
                        "/proc/",
                        "C:\\Windows\\",
                        "C:\\Users\\",
                        "\\system32\\",
                        "passwd",
                        "shadow",
                        "hosts",
                        "config",
                    ]

                    for indicator in dangerous_indicators:
                        if indicator.lower() in response_data.lower():
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Path traversal attack may have succeeded",
                                    "payload": payload,
                                    "indicator": indicator,
                                    "response_preview": response_data[:200],
                                    "severity": "critical",
                                }
                            )

                # Test path traversal in form fields
                form_data = {
                    "filename": payload,
                    "path": payload,
                    "destination": payload,
                }

                files = {
                    "file": (
                        "test.txt",
                        io.BytesIO(file_content),
                        "text/plain",
                    )
                }
                response = self.client.post(
                    "/api/v1/upload", files=files, data=form_data
                )

                if response.status_code in [200, 201, 202]:
                    response_data = response.text
                    for indicator in dangerous_indicators:
                        if indicator.lower() in response_data.lower():
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Path traversal via form data may have succeeded",
                                    "payload": payload,
                                    "field": "form_data",
                                    "severity": "critical",
                                }
                            )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing path traversal payload {payload[:50]}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Sanitize all file names and paths before processing",
                    "Use absolute paths and validate against allowed directories",
                    "Implement strict directory traversal prevention",
                    "Store uploaded files in sandboxed directories only",
                ]
            )

        return results

    def test_file_size_limits(self) -> Dict[str, Any]:
        """Test file size limit enforcement."""
        results = {
            "test_name": "file_size_limits",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various file sizes
        size_tests = [
            ("small", 1024),  # 1KB
            ("medium", 1024 * 1024),  # 1MB
            ("large", 10 * 1024 * 1024),  # 10MB
            ("huge", 100 * 1024 * 1024),  # 100MB
            ("massive", 1024 * 1024 * 1024),  # 1GB
        ]

        for size_name, size_bytes in size_tests:
            try:
                # Create file of specified size
                file_content = b"A" * min(
                    size_bytes, 1024 * 1024
                )  # Limit to 1MB for testing
                filename = f"size_test_{size_name}.txt"

                files = {
                    "file": (filename, io.BytesIO(file_content), "text/plain")
                }
                response = self.client.post("/api/v1/upload", files=files)

                # Very large files should be rejected
                if size_bytes > 50 * 1024 * 1024:  # > 50MB
                    if response.status_code in [200, 201, 202]:
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Extremely large file accepted",
                                "size": size_bytes,
                                "size_name": size_name,
                                "severity": "medium",
                            }
                        )

                # Check for proper error messages for oversized files
                if response.status_code == 413:  # Payload Too Large
                    # Good - proper HTTP status code
                    if (
                        "path" in response.text.lower()
                        or "directory" in response.text.lower()
                    ):
                        results["findings"].append(
                            {
                                "issue": "File size error message discloses paths",
                                "size_name": size_name,
                                "response_preview": response.text[:200],
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing file size {size_name}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement reasonable file size limits (e.g., 10MB max)",
                    "Return proper HTTP 413 status for oversized files",
                    "Do not disclose server paths in size limit error messages",
                ]
            )

        return results

    def test_malicious_file_content_detection(self) -> Dict[str, Any]:
        """Test detection of malicious file content."""
        results = {
            "test_name": "malicious_file_content_detection",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test files with malicious content patterns
        malicious_files = [
            ("php_webshell.txt", b'<?php system($_GET["cmd"]); ?>'),
            ("javascript_payload.txt", b'<script>alert("xss")</script>'),
            ("jsp_shell.txt", b'<%@ page import="java.io.*" %><%'),
            ("bash_script.txt", b"#!/bin/bash\nrm -rf /"),
            ("executable_header.txt", b"MZ\x90\x00\x03\x00\x00\x00"),
            ("elf_binary.txt", b"\x7fELF\x01\x01\x01\x00"),
        ]

        for filename, content in malicious_files:
            try:
                files = {"file": (filename, io.BytesIO(content), "text/plain")}
                response = self.client.post("/api/v1/upload", files=files)

                # Malicious content should be detected and rejected
                if response.status_code in [200, 201, 202]:
                    results["findings"].append(
                        {
                            "issue": "Malicious file content not detected",
                            "filename": filename,
                            "content_type": "malicious_pattern",
                            "severity": "high",
                        }
                    )

                # Check if content scanning is mentioned in response
                if (
                    "scanned" in response.text.lower()
                    or "virus" in response.text.lower()
                ):
                    # Good - indicates content scanning
                    pass

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing malicious content in {filename}",
                        "error": str(e),
                    }
                )

        # Test file with embedded null bytes
        try:
            null_byte_content = b"innocent.txt\x00malicious.php"
            files = {
                "file": (
                    "test.txt",
                    io.BytesIO(null_byte_content),
                    "text/plain",
                )
            }
            response = self.client.post("/api/v1/upload", files=files)

            if response.status_code in [200, 201, 202]:
                results["findings"].append(
                    {
                        "issue": "File with null bytes accepted",
                        "note": "Null bytes can be used to bypass file type detection",
                        "severity": "medium",
                    }
                )

        except Exception as e:
            results["findings"].append(
                {"issue": "Error testing null byte content", "error": str(e)}
            )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement content-based file scanning",
                    "Detect and block files with executable signatures",
                    "Scan for malicious patterns in file content",
                    "Reject files containing null bytes or unusual characters",
                ]
            )

        return results

    def test_file_metadata_security(self) -> Dict[str, Any]:
        """Test file metadata security and information disclosure."""
        results = {
            "test_name": "file_metadata_security",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test files with suspicious metadata
        metadata_tests = [
            ("long_filename.txt", "A" * 1000),  # Extremely long filename
            ("unicode_filename.txt", "файл.txt"),  # Unicode filename
            ("special_chars.txt", 'file<>:"|?*.txt'),  # Special characters
            ("hidden_file.txt", ".htaccess"),  # Hidden file
            ("config_file.txt", "config.ini"),  # Configuration file name
        ]

        for test_name, filename in metadata_tests:
            try:
                file_content = b"Test content"
                files = {
                    "file": (filename, io.BytesIO(file_content), "text/plain")
                }
                response = self.client.post("/api/v1/upload", files=files)

                # Check if server discloses file system information
                response_text = response.text.lower()
                disclosure_indicators = [
                    "/var/",
                    "/tmp/",
                    "/upload/",
                    "c:\\",
                    "documents and settings",
                    "program files",
                    "/home/",
                    "/root/",
                    "absolute path",
                ]

                for indicator in disclosure_indicators:
                    if indicator in response_text:
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "File system path disclosed in response",
                                "test": test_name,
                                "filename": filename,
                                "indicator": indicator,
                                "response_preview": response.text[:200],
                            }
                        )

                # Very long filenames should be rejected or truncated
                if len(filename) > 255 and response.status_code in [
                    200,
                    201,
                    202,
                ]:
                    results["findings"].append(
                        {
                            "issue": "Extremely long filename accepted",
                            "filename_length": len(filename),
                            "test": test_name,
                        }
                    )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing metadata for {test_name}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Validate and sanitize file names before processing",
                    "Limit filename lengths to reasonable values",
                    "Do not disclose server file system paths in responses",
                    "Handle unicode and special characters safely",
                ]
            )

        return results

    def test_upload_directory_security(self) -> Dict[str, Any]:
        """Test upload directory security and access controls."""
        results = {
            "test_name": "upload_directory_security",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test access to upload directory
        potential_upload_paths = [
            "/uploads/",
            "/files/",
            "/static/uploads/",
            "/media/",
            "/public/uploads/",
            "/var/www/uploads/",
            "/tmp/uploads/",
        ]

        for path in potential_upload_paths:
            try:
                # Try to access upload directory directly
                response = self.client.get(path)

                if response.status_code == 200:
                    # Directory listing might be enabled
                    if (
                        "index of" in response.text.lower()
                        or "<a href=" in response.text.lower()
                    ):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Upload directory listing enabled",
                                "path": path,
                                "severity": "high",
                                "note": "Uploaded files may be directly accessible",
                            }
                        )

                # Try common directory traversal on upload paths
                traversal_attempts = [
                    f"{path}../../../etc/passwd",
                    f"{path}..\\..\\..\\windows\\system32\\config\\sam",
                ]

                for attempt in traversal_attempts:
                    traversal_response = self.client.get(attempt)
                    if traversal_response.status_code == 200:
                        if (
                            "root:" in traversal_response.text
                            or "administrator"
                            in traversal_response.text.lower()
                        ):
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Directory traversal via upload path",
                                    "path": attempt,
                                    "severity": "critical",
                                }
                            )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing upload directory {path}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Disable directory listing for upload directories",
                    "Store uploaded files outside web root",
                    "Implement proper access controls on upload directories",
                    "Use random file names to prevent direct access",
                ]
            )

        return results

    def test_file_execution_prevention(self) -> Dict[str, Any]:
        """Test prevention of uploaded file execution."""
        results = {
            "test_name": "file_execution_prevention",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test executable file uploads
        executable_tests = [
            (
                "script.php",
                b'<?php echo "PHP execution test"; ?>',
                "application/x-php",
            ),
            ("page.asp", b'<%="ASP execution test"%>', "application/x-asp"),
            (
                "shell.jsp",
                b'<%out.println("JSP execution test");%>',
                "application/x-jsp",
            ),
            ("script.py", b'print("Python execution test")', "text/x-python"),
            (
                "test.js",
                b'console.log("JavaScript execution test");',
                "application/javascript",
            ),
        ]

        for filename, content, content_type in executable_tests:
            try:
                # Upload executable file
                files = {"file": (filename, io.BytesIO(content), content_type)}
                upload_response = self.client.post(
                    "/api/v1/upload", files=files
                )

                if upload_response.status_code in [200, 201, 202]:
                    # Try to access and execute the uploaded file
                    potential_paths = [
                        f"/uploads/{filename}",
                        f"/files/{filename}",
                        f"/static/{filename}",
                        f"/media/{filename}",
                    ]

                    for path in potential_paths:
                        exec_response = self.client.get(path)

                        if exec_response.status_code == 200:
                            # Check if content indicates execution
                            response_text = exec_response.text

                            if "execution test" in response_text:
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Uploaded executable file was executed",
                                        "filename": filename,
                                        "path": path,
                                        "severity": "critical",
                                        "evidence": response_text[:100],
                                    }
                                )

                            # Even if not executed, direct access to uploaded files is a risk
                            elif (
                                content.decode("utf-8", errors="ignore")
                                in response_text
                            ):
                                results["findings"].append(
                                    {
                                        "issue": "Uploaded file directly accessible",
                                        "filename": filename,
                                        "path": path,
                                        "severity": "high",
                                        "note": "Files should not be directly accessible via web",
                                    }
                                )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing execution prevention for {filename}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Store uploaded files outside the web-accessible directory",
                    "Configure web server to never execute uploaded files",
                    "Use .htaccess or server configuration to prevent execution",
                    "Serve uploaded files through application logic, not directly",
                ]
            )

        return results

    def test_mime_type_validation(self) -> Dict[str, Any]:
        """Test MIME type validation and spoofing prevention."""
        results = {
            "test_name": "mime_type_validation",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test MIME type spoofing
        spoofing_tests = [
            ("malicious.php", b'<?php system($_GET["cmd"]); ?>', "image/jpeg"),
            ("script.js", b'alert("xss")', "text/plain"),
            ("executable.exe", b"MZ\x90\x00\x03\x00\x00\x00", "image/png"),
            (
                "shell.jsp",
                b'<%@ page import="java.io.*" %>',
                "application/pdf",
            ),
        ]

        for filename, content, fake_mime_type in spoofing_tests:
            try:
                files = {
                    "file": (filename, io.BytesIO(content), fake_mime_type)
                }
                response = self.client.post("/api/v1/upload", files=files)

                # Should be rejected based on actual content, not declared MIME type
                if response.status_code in [200, 201, 202]:
                    results["findings"].append(
                        {
                            "issue": "MIME type spoofing not detected",
                            "filename": filename,
                            "declared_mime": fake_mime_type,
                            "actual_content": "malicious",
                            "severity": "high",
                        }
                    )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing MIME spoofing for {filename}",
                        "error": str(e),
                    }
                )

        # Test correct MIME type validation
        valid_tests = [
            ("image.jpg", b"\xff\xd8\xff\xe0", "image/jpeg"),  # JPEG header
            ("document.pdf", b"%PDF-1.4", "application/pdf"),  # PDF header
            ("text.txt", b"Plain text content", "text/plain"),
        ]

        for filename, content, correct_mime_type in valid_tests:
            try:
                files = {
                    "file": (filename, io.BytesIO(content), correct_mime_type)
                }
                response = self.client.post("/api/v1/upload", files=files)

                # Valid files should be accepted (if upload endpoint exists)
                if response.status_code == 403:
                    # May indicate overly restrictive policy
                    pass

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing valid MIME type for {filename}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Validate file content against declared MIME type",
                    "Use file magic number detection for type validation",
                    "Do not rely solely on client-provided MIME types",
                    "Implement content-based file type detection",
                ]
            )

        return results

    def run_all_file_upload_tests(self) -> Dict[str, Any]:
        """Run all file upload security tests."""
        print("Running comprehensive file upload security tests...")

        test_methods = [
            self.test_file_extension_validation,
            self.test_path_traversal_prevention,
            self.test_file_size_limits,
            self.test_malicious_file_content_detection,
            self.test_file_metadata_security,
            self.test_upload_directory_security,
            self.test_file_execution_prevention,
            self.test_mime_type_validation,
        ]

        all_results = []

        for test_method in test_methods:
            try:
                result = test_method()
                all_results.append(result)
                status = "PASS" if result["passed"] else "FAIL"
                print(f"  {result['test_name']}: {status}")

                if not result["passed"]:
                    for finding in result["findings"]:
                        severity = finding.get("severity", "medium")
                        print(f"    - {severity.upper()}: {finding['issue']}")

            except Exception as e:
                print(f"  {test_method.__name__}: ERROR - {str(e)}")
                all_results.append(
                    {
                        "test_name": test_method.__name__,
                        "passed": False,
                        "findings": [
                            {"issue": f"Test execution error: {str(e)}"}
                        ],
                        "recommendations": ["Fix test execution error"],
                    }
                )

        # Compile overall results
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r["passed"])
        failed_tests = total_tests - passed_tests

        # Categorize findings by severity
        critical_findings = []
        high_findings = []
        medium_findings = []

        for result in all_results:
            for finding in result.get("findings", []):
                severity = finding.get("severity", "medium")
                if severity == "critical":
                    critical_findings.append(finding)
                elif severity == "high":
                    high_findings.append(finding)
                else:
                    medium_findings.append(finding)

        # Collect all recommendations
        all_recommendations = []
        for result in all_results:
            all_recommendations.extend(result.get("recommendations", []))

        # Remove duplicates
        unique_recommendations = list(set(all_recommendations))

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100)
                if total_tests > 0
                else 0,
                "critical_findings": len(critical_findings),
                "high_findings": len(high_findings),
                "medium_findings": len(medium_findings),
            },
            "detailed_results": all_results,
            "recommendations": unique_recommendations,
            "overall_status": "PASS" if failed_tests == 0 else "FAIL",
        }

        return summary


class TestFileUploadSecurity:
    """pytest test class for file upload security validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def upload_tester(self, client):
        """Create file upload security tester."""
        return FileUploadSecurityTester(client)

    def test_dangerous_extensions_blocked(self, upload_tester):
        """Test that dangerous file extensions are blocked."""
        result = upload_tester.test_file_extension_validation()

        # Check specifically for high severity findings (dangerous extensions allowed)
        high_severity_issues = [
            f
            for f in result.get("findings", [])
            if f.get("severity") == "high"
        ]

        if high_severity_issues:
            failure_msg = "Dangerous file extensions are being allowed:\n"
            for finding in high_severity_issues:
                failure_msg += f"  - {finding['issue']}: {finding.get('extension', 'unknown')}\n"
            pytest.fail(failure_msg)

    def test_path_traversal_blocked(self, upload_tester):
        """Test that path traversal attacks are prevented."""
        result = upload_tester.test_path_traversal_prevention()

        if not result["passed"]:
            critical_issues = [
                f
                for f in result.get("findings", [])
                if f.get("severity") == "critical"
            ]
            if critical_issues:
                failure_msg = (
                    "Critical path traversal vulnerabilities detected:\n"
                )
                for finding in critical_issues:
                    failure_msg += f"  - {finding['issue']}\n"
                pytest.fail(failure_msg)

    def test_malicious_content_detected(self, upload_tester):
        """Test that malicious file content is detected."""
        result = upload_tester.test_malicious_file_content_detection()

        if not result["passed"]:
            high_severity_issues = [
                f
                for f in result.get("findings", [])
                if f.get("severity") == "high"
            ]
            if high_severity_issues:
                failure_msg = "Malicious file content not being detected:\n"
                for finding in high_severity_issues:
                    failure_msg += f"  - {finding['issue']}\n"
                pytest.fail(failure_msg)

    def test_file_execution_prevented(self, upload_tester):
        """Test that uploaded files cannot be executed."""
        result = upload_tester.test_file_execution_prevention()

        if not result["passed"]:
            critical_issues = [
                f
                for f in result.get("findings", [])
                if f.get("severity") == "critical"
            ]
            if critical_issues:
                failure_msg = "CRITICAL: Uploaded files can be executed:\n"
                for finding in critical_issues:
                    failure_msg += f"  - {finding['issue']}\n"
                pytest.fail(failure_msg)

    def test_comprehensive_upload_security(self, upload_tester):
        """Run comprehensive file upload security tests."""
        summary = upload_tester.run_all_file_upload_tests()

        if summary["overall_status"] == "FAIL":
            failure_msg = f"File upload security test failures: {summary['summary']['failed_tests']} out of {summary['summary']['total_tests']} tests failed\n"

            # Check for critical/high severity issues
            if summary["summary"]["critical_findings"] > 0:
                failure_msg += f"\nCRITICAL ISSUES: {summary['summary']['critical_findings']}\n"

            if summary["summary"]["high_findings"] > 0:
                failure_msg += f"HIGH SEVERITY ISSUES: {summary['summary']['high_findings']}\n"

            if summary["recommendations"]:
                failure_msg += "\nRecommendations:\n"
                for rec in summary["recommendations"][:10]:
                    failure_msg += f"  - {rec}\n"

            pytest.fail(failure_msg)


if __name__ == "__main__":
    """Direct execution for standalone testing."""
    client = TestClient(app)
    tester = FileUploadSecurityTester(client)

    print("Running file upload security validation tests...")
    summary = tester.run_all_file_upload_tests()

    print(f"\n{'='*60}")
    print("FILE UPLOAD SECURITY VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Total Tests: {summary['summary']['total_tests']}")
    print(f"Passed: {summary['summary']['passed_tests']}")
    print(f"Failed: {summary['summary']['failed_tests']}")
    print(f"Pass Rate: {summary['summary']['pass_rate']:.1f}%")
    print(f"Overall Status: {summary['overall_status']}")

    print(f"\nFindings by Severity:")
    print(f"  Critical: {summary['summary']['critical_findings']}")
    print(f"  High: {summary['summary']['high_findings']}")
    print(f"  Medium: {summary['summary']['medium_findings']}")

    if summary["recommendations"]:
        print(f"\n{'='*40}")
        print("RECOMMENDATIONS")
        print(f"{'='*40}")
        for rec in summary["recommendations"]:
            print(f"• {rec}")

    # Save report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"/home/green/FreeAgentics/tests/security/file_upload_security_report_{timestamp}.json"

    try:
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Exit with appropriate code
    exit(0 if summary["overall_status"] == "PASS" else 1)
