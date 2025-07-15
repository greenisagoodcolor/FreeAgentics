"""
OWASP Top 10 Validation Test Suite
Task 14.11 - Agent 4 - Security Validation Tests

Test-driven approach to validate OWASP Top 10 security implementations.
"""

import re
from pathlib import Path

import pytest


class TestOWASPValidation:
    """Test suite for validating OWASP Top 10 security implementations."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def api_files(self, project_root):
        """Get all API files."""
        api_files = []
        for pattern in ["api/**/*.py", "auth/**/*.py"]:
            api_files.extend(project_root.glob(pattern))
        return [f for f in api_files if self._is_application_file(f)]

    def _is_application_file(self, file_path: Path) -> bool:
        """Check if file is application code (not test or third-party)."""
        exclude_dirs = {"venv", "node_modules", "__pycache__", ".git"}
        for part in file_path.parts:
            if part in exclude_dirs:
                return False
        return True

    def _get_file_content(self, file_path: Path) -> str:
        """Get file content safely."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def test_a01_api_endpoints_have_authentication(self, api_files):
        """Test A01: Verify all API endpoints have authentication."""
        unprotected_endpoints = []

        for file_path in api_files:
            content = self._get_file_content(file_path)

            # Find FastAPI endpoints
            endpoint_patterns = [
                r"@app\.(get|post|put|delete|patch)\s*\([^)]*\)",
                r"@router\.(get|post|put|delete|patch)\s*\([^)]*\)",
            ]

            for pattern in endpoint_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip public endpoints
                    if any(
                        public in match.group(0)
                        for public in ["/health", "/docs", "/redoc", "/openapi", "/favicon"]
                    ):
                        continue

                    line_number = content[: match.start()].count("\n") + 1
                    lines = content.split("\n")

                    # Check for authentication in surrounding lines
                    protected = False
                    for i in range(max(0, line_number - 5), min(len(lines), line_number + 10)):
                        if any(
                            auth in lines[i]
                            for auth in [
                                "@require_permission",
                                "@require_role",
                                "get_current_user",
                                "Depends(get_current_user)",
                                "require_permission",
                            ]
                        ):
                            protected = True
                            break

                    if not protected:
                        unprotected_endpoints.append(
                            {
                                "file": str(file_path),
                                "line": line_number,
                                "endpoint": match.group(0),
                            }
                        )

        # Assert that all endpoints are protected
        assert (
            len(unprotected_endpoints) == 0
        ), f"Found {len(unprotected_endpoints)} unprotected endpoints:\n" + "\n".join(
            f"  {ep['file']}:{ep['line']} - {ep['endpoint']}" for ep in unprotected_endpoints
        )

    def test_a02_no_hardcoded_secrets(self, project_root):
        """Test A02: Verify no hardcoded secrets in code."""
        secret_findings = []

        # Check application files
        app_files = []
        for pattern in ["api/**/*.py", "auth/**/*.py", "agents/**/*.py", "config/**/*.py"]:
            app_files.extend(project_root.glob(pattern))

        app_files.extend(project_root.glob("*.py"))  # Root level files

        secret_patterns = [
            (r'secret\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded secret"),
            (r'key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded key"),
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token"),
        ]

        for file_path in app_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            for pattern, description in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip obvious placeholders
                    if any(
                        placeholder in match.group(0).lower()
                        for placeholder in [
                            "your_",
                            "example",
                            "placeholder",
                            "test_",
                            "dummy",
                            "change_me",
                        ]
                    ):
                        continue

                    line_number = content[: match.start()].count("\n") + 1
                    secret_findings.append(
                        {
                            "file": str(file_path),
                            "line": line_number,
                            "type": description,
                            "evidence": match.group(0),
                        }
                    )

        assert (
            len(secret_findings) == 0
        ), f"Found {len(secret_findings)} hardcoded secrets:\n" + "\n".join(
            f"  {s['file']}:{s['line']} - {s['type']}: {s['evidence']}" for s in secret_findings
        )

    def test_a03_no_sql_injection_patterns(self, project_root):
        """Test A03: Verify no SQL injection vulnerabilities."""
        injection_findings = []

        # Check database and API files
        db_files = []
        for pattern in ["database/**/*.py", "api/**/*.py"]:
            db_files.extend(project_root.glob(pattern))

        injection_patterns = [
            (r"cursor\.execute\([^)]*%[^)]*\)", "SQL injection via string formatting"),
            (r"\.query\([^)]*%[^)]*\)", "SQL injection via query formatting"),
            (r"SELECT.*\+.*FROM", "SQL injection via string concatenation"),
            (r'f"SELECT.*\{[^}]*\}.*"', "SQL injection via f-string"),
        ]

        for file_path in db_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            for pattern, description in injection_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_number = content[: match.start()].count("\n") + 1
                    injection_findings.append(
                        {
                            "file": str(file_path),
                            "line": line_number,
                            "type": description,
                            "evidence": match.group(0),
                        }
                    )

        assert len(injection_findings) == 0, (
            f"Found {len(injection_findings)} potential SQL injection vulnerabilities:\n"
            + "\n".join(
                f"  {i['file']}:{i['line']} - {i['type']}: {i['evidence']}"
                for i in injection_findings
            )
        )

    def test_a04_rate_limiting_implemented(self, project_root):
        """Test A04: Verify rate limiting is implemented."""
        rate_limit_files = list(project_root.glob("**/rate_limit*.py"))
        rate_limit_files.extend(project_root.glob("**/*rate*limit*.py"))

        # Filter application files
        app_rate_limit_files = [f for f in rate_limit_files if self._is_application_file(f)]

        assert (
            len(app_rate_limit_files) > 0
        ), "No rate limiting implementation found in application code"

        # Check that rate limiting is actually used
        api_files = list(project_root.glob("api/**/*.py"))
        rate_limit_usage = False

        for file_path in api_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            if re.search(r"rate_limit|RateLimit|@rate_limit", content):
                rate_limit_usage = True
                break

        assert rate_limit_usage, "Rate limiting implementation found but not used in API endpoints"

    def test_a05_debug_mode_disabled(self, project_root):
        """Test A05: Verify debug mode is properly configured."""
        debug_issues = []

        # Check configuration files
        config_files = []
        config_files.extend(project_root.glob("*.py"))
        config_files.extend(project_root.glob("config/**/*.py"))

        for file_path in config_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            # Look for hardcoded DEBUG = True
            if re.search(r"DEBUG\s*=\s*True(?!\s*#.*test)", content):
                line_number = content.find("DEBUG = True")
                if line_number != -1:
                    line_number = content[:line_number].count("\n") + 1
                    debug_issues.append(
                        {"file": str(file_path), "line": line_number, "issue": "DEBUG = True found"}
                    )

        assert (
            len(debug_issues) == 0
        ), f"Found {len(debug_issues)} debug mode issues:\n" + "\n".join(
            f"  {d['file']}:{d['line']} - {d['issue']}" for d in debug_issues
        )

    def test_a06_dependencies_tracked(self, project_root):
        """Test A06: Verify dependencies are properly tracked."""
        required_files = [
            "requirements.txt",
            "requirements-core.txt",
            "requirements-dev.txt",
            "requirements-production.txt",
        ]

        found_files = []
        for req_file in required_files:
            if (project_root / req_file).exists():
                found_files.append(req_file)

        assert (
            len(found_files) >= 2
        ), f"Insufficient dependency files found. Expected at least 2, found {len(found_files)}: {found_files}"

        # Check for package.json
        package_json = project_root / "web" / "package.json"
        if package_json.exists():
            # Frontend dependencies are tracked
            pass

    def test_a07_secure_password_hashing(self, project_root):
        """Test A07: Verify secure password hashing is implemented."""
        auth_files = []
        auth_files.extend(project_root.glob("auth/**/*.py"))
        auth_files.extend(project_root.glob("api/**/auth*.py"))

        secure_hashing_found = False
        weak_hashing_found = []

        for file_path in auth_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            # Check for secure hashing
            if re.search(r"bcrypt|scrypt|argon2|PBKDF2", content):
                secure_hashing_found = True

            # Check for weak hashing
            weak_patterns = [
                r"hashlib\.md5",
                r"hashlib\.sha1",
                r"hashlib\.sha256\([^)]*password[^)]*\)",
            ]

            for pattern in weak_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_number = content[: match.start()].count("\n") + 1
                    weak_hashing_found.append(
                        {"file": str(file_path), "line": line_number, "pattern": match.group(0)}
                    )

        assert (
            secure_hashing_found
        ), "No secure password hashing implementation found (bcrypt, scrypt, argon2, PBKDF2)"

        assert (
            len(weak_hashing_found) == 0
        ), f"Found {len(weak_hashing_found)} weak password hashing patterns:\n" + "\n".join(
            f"  {w['file']}:{w['line']} - {w['pattern']}" for w in weak_hashing_found
        )

    def test_a08_no_unsafe_deserialization(self, project_root):
        """Test A08: Verify no unsafe deserialization patterns."""
        unsafe_patterns = [
            (r"pickle\.load\(", "Unsafe pickle deserialization"),
            (r"pickle\.loads\(", "Unsafe pickle deserialization"),
            (r"eval\(", "Unsafe eval usage"),
            (r"exec\(", "Unsafe exec usage"),
        ]

        app_files = []
        for pattern in ["api/**/*.py", "auth/**/*.py", "agents/**/*.py", "database/**/*.py"]:
            app_files.extend(project_root.glob(pattern))

        unsafe_findings = []

        for file_path in app_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            for pattern, description in unsafe_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_number = content[: match.start()].count("\n") + 1
                    unsafe_findings.append(
                        {
                            "file": str(file_path),
                            "line": line_number,
                            "type": description,
                            "evidence": match.group(0),
                        }
                    )

        assert (
            len(unsafe_findings) == 0
        ), f"Found {len(unsafe_findings)} unsafe deserialization patterns:\n" + "\n".join(
            f"  {u['file']}:{u['line']} - {u['type']}: {u['evidence']}" for u in unsafe_findings
        )

    def test_a09_security_logging_implemented(self, project_root):
        """Test A09: Verify security logging is implemented."""
        # Check for security logging files
        security_log_files = []
        security_log_files.extend(project_root.glob("**/security_log*.py"))
        security_log_files.extend(project_root.glob("**/audit*.py"))
        security_log_files.extend(project_root.glob("observability/**/*.py"))

        app_log_files = [f for f in security_log_files if self._is_application_file(f)]

        assert len(app_log_files) > 0, "No security logging implementation found"

        # Check for monitoring endpoints
        api_files = list(project_root.glob("api/**/*.py"))
        monitoring_endpoints = False

        for file_path in api_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            if any(
                pattern in content for pattern in ["/health", "/metrics", "/monitoring", "/status"]
            ):
                monitoring_endpoints = True
                break

        assert monitoring_endpoints, "No monitoring endpoints found in API"

    def test_a10_no_ssrf_vulnerabilities(self, project_root):
        """Test A10: Verify no SSRF vulnerabilities."""
        app_files = []
        for pattern in ["api/**/*.py", "agents/**/*.py", "knowledge_graph/**/*.py"]:
            app_files.extend(project_root.glob(pattern))

        ssrf_findings = []

        url_patterns = [
            r"requests\.get\(",
            r"requests\.post\(",
            r"httpx\.get\(",
            r"httpx\.post\(",
        ]

        for file_path in app_files:
            if not self._is_application_file(file_path):
                continue

            content = self._get_file_content(file_path)

            for pattern in url_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_number = content[: match.start()].count("\n") + 1
                    lines = content.split("\n")

                    if line_number <= len(lines):
                        line_content = lines[line_number - 1]

                        # Check if URL might be user-controlled
                        if any(
                            keyword in line_content.lower()
                            for keyword in ["request.", "user", "input", "param", "args", "form"]
                        ):
                            ssrf_findings.append(
                                {
                                    "file": str(file_path),
                                    "line": line_number,
                                    "evidence": line_content.strip(),
                                }
                            )

        assert (
            len(ssrf_findings) == 0
        ), f"Found {len(ssrf_findings)} potential SSRF vulnerabilities:\n" + "\n".join(
            f"  {s['file']}:{s['line']} - {s['evidence']}" for s in ssrf_findings
        )

    def test_security_assessment_report_exists(self, project_root):
        """Test that security assessment report exists and is current."""
        assessment_files = [
            "security/OWASP_TOP_10_ASSESSMENT.md",
            "security/OWASP_TOP_10_ASSESSMENT_UPDATED.md",
            "security/owasp_focused_assessment_report.json",
        ]

        existing_files = []
        for assessment_file in assessment_files:
            if (project_root / assessment_file).exists():
                existing_files.append(assessment_file)

        assert (
            len(existing_files) >= 1
        ), f"No security assessment reports found. Expected at least one of: {assessment_files}"

    def test_security_tools_available(self, project_root):
        """Test that security assessment tools are available."""
        security_tools = [
            "security/owasp_assessment.py",
            "security/owasp_assessment_focused.py",
            "security/owasp_assessment_2024_comprehensive.py",
        ]

        existing_tools = []
        for tool in security_tools:
            if (project_root / tool).exists():
                existing_tools.append(tool)

        assert (
            len(existing_tools) >= 1
        ), f"No security assessment tools found. Expected at least one of: {security_tools}"


class TestSecurityCompliance:
    """Extended tests for security compliance validation."""

    def test_input_validation_framework(self, project_root):
        """Test that input validation framework is implemented."""
        api_files = list(project_root.glob("api/**/*.py"))

        validation_found = False
        validation_files = []

        for file_path in api_files:
            try:
                content = self._get_file_content(file_path)

                # Check for Pydantic or validation patterns
                if any(
                    pattern in content
                    for pattern in [
                        "from pydantic import",
                        "pydantic.BaseModel",
                        "validate_",
                        "ValidationError",
                    ]
                ):
                    validation_found = True
                    validation_files.append(str(file_path))
            except Exception:
                continue

        assert validation_found, "No input validation framework found in API files"

        assert len(validation_files) > 0, "Input validation framework not used in API endpoints"

    def _get_file_content(self, file_path):
        """Helper to get file content."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
