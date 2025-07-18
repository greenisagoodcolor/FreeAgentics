"""
Path Traversal Prevention Testing

Comprehensive tests for validating path traversal attack prevention
across all application components and endpoints.

This module focuses on:
1. File access path traversal prevention
2. API parameter path traversal testing
3. URL path manipulation testing
4. Template inclusion path traversal
5. Configuration file access prevention
6. Log file access prevention
7. Static file serving security
8. Directory listing prevention
9. Symbolic link exploitation prevention
10. OS-specific path traversal variants
"""

import json
import time
import urllib.parse
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.main import app


class PathTraversalPreventionTester:
    """Comprehensive path traversal prevention tester."""

    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []

        # Comprehensive path traversal payloads
        self.path_traversal_payloads = {
            "basic_unix": [
                "../../../etc/passwd",
                "../../../etc/shadow",
                "../../../var/log/auth.log",
                "../../../proc/self/environ",
                "../../../home/user/.ssh/id_rsa",
                "../../../root/.bash_history",
                "../../../../etc/passwd",
                "../../../../../etc/passwd",
                "../../../../../../etc/passwd",
            ],
            "basic_windows": [
                "..\\..\\..\\windows\\system32\\config\\sam",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "..\\..\\..\\users\\administrator\\ntuser.dat",
                "..\\..\\..\\boot.ini",
                "..\\..\\..\\autoexec.bat",
                "..\\..\\..\\config.sys",
            ],
            "url_encoded": [
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "%2e%2e%5c%2e%2e%5c%2e%2e%5cwindows%5csystem32%5cconfig%5csam",
                "%252e%252e%252fetc%252fpasswd",
                "%252e%252e%255cetc%255cpasswd",
            ],
            "double_encoded": [
                "%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd",
                "%25252e%25252e%25252fetc%25252fpasswd",
            ],
            "unicode_encoded": [
                "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
                "..%e0%80%af..%e0%80%afetc%e0%80%afpasswd",
                "\u002e\u002e\u002f\u002e\u002e\u002f\u002e\u002e\u002fetc\u002fpasswd",
            ],
            "null_byte": [
                "../../../etc/passwd\x00.txt",
                "../../../etc/shadow\x00.jpg",
                "../../../windows/system32/config/sam\x00.png",
            ],
            "mixed_separators": [
                "..\\../..\\../..\\../etc/passwd",
                "../..\\../..\\../etc/passwd",
                "..\\\\..//..\\\\..//etc//passwd",
            ],
            "absolute_paths": [
                "/etc/passwd",
                "/etc/shadow",
                "/var/log/auth.log",
                "C:\\Windows\\System32\\config\\SAM",
                "C:\\boot.ini",
                "C:\\Users\\Administrator\\NTUSER.DAT",
            ],
            "file_protocol": [
                "file:///etc/passwd",
                "file:///c:/windows/system32/config/sam",
                "file://localhost/etc/passwd",
            ],
            "multiple_slashes": [
                "....//....//....//etc/passwd",
                "....\\\\....\\\\....\\\\windows\\\\system32\\\\config\\\\sam",
                "...//...//.../etc/passwd",
            ],
            "bypass_attempts": [
                "....//....//etc/passwd",
                ".\\.\\.\\.\\etc\\passwd",
                ".\\../.\\../etc/passwd",
                "..//.//..//.//..//etc/passwd",
                "path/../../etc/passwd",
                "file.txt/../../../etc/passwd",
            ],
        }

        # Sensitive files that should never be accessible
        self.sensitive_files = [
            "/etc/passwd",
            "/etc/shadow",
            "/etc/hosts",
            "/etc/group",
            "/proc/self/environ",
            "/proc/version",
            "/proc/cmdline",
            "/var/log/auth.log",
            "/var/log/syslog",
            "/var/log/messages",
            "/home/user/.ssh/id_rsa",
            "/home/user/.bash_history",
            "/root/.bash_history",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM",
            "C:\\Windows\\System32\\config\\SECURITY",
            "C:\\Windows\\System32\\config\\SOFTWARE",
            "C:\\Windows\\System32\\config\\SYSTEM",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "C:\\boot.ini",
            "C:\\autoexec.bat",
            "C:\\config.sys",
            "C:\\Users\\Administrator\\NTUSER.DAT",
            "C:\\Documents and Settings\\Administrator\\NTUSER.DAT",
        ]

        # Application configuration files
        self.config_files = [
            ".env",
            "config.ini",
            "settings.json",
            "database.yml",
            "secrets.txt",
            "private.key",
            "id_rsa",
            "credentials.json",
            "application.properties",
            "web.config",
            "httpd.conf",
            "nginx.conf",
            "apache2.conf",
            ".htaccess",
            ".htpasswd",
        ]

    def test_api_parameter_path_traversal(self) -> Dict[str, Any]:
        """Test path traversal in API parameters."""
        results = {
            "test_name": "api_parameter_path_traversal",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various API endpoints that might accept file/path parameters
        api_endpoints = [
            ("/api/v1/files/{path}", "path"),
            ("/api/v1/download", "filename"),
            ("/api/v1/static", "file"),
            ("/api/v1/logs", "logfile"),
            ("/api/v1/config", "config_file"),
            ("/api/v1/backup", "backup_path"),
            ("/api/v1/export", "export_path"),
            ("/api/v1/import", "import_file"),
        ]

        for endpoint_template, param_name in api_endpoints:
            for category, payloads in self.path_traversal_payloads.items():
                for payload in payloads[
                    :3
                ]:  # Test first 3 payloads per category
                    try:
                        # Test in URL path
                        if "{path}" in endpoint_template:
                            endpoint = endpoint_template.replace(
                                "{path}", urllib.parse.quote(payload, safe="")
                            )
                            response = self.client.get(endpoint)
                        else:
                            # Test in query parameters
                            endpoint = endpoint_template
                            params = {param_name: payload}
                            response = self.client.get(endpoint, params=params)

                        # Check for successful traversal indicators
                        if response.status_code == 200:
                            response_text = response.text.lower()

                            # Check for sensitive file content
                            sensitive_indicators = [
                                "root:",
                                "administrator:",
                                "daemon:",
                                "sys:",
                                "[boot loader]",
                                "[operating systems]",
                                "password for",
                                "secret",
                                "private key",
                                "-----begin",
                                "-----end",
                                "ssh-rsa",
                                "ssh-dss",
                            ]

                            for indicator in sensitive_indicators:
                                if indicator in response_text:
                                    results["passed"] = False
                                    results["findings"].append(
                                        {
                                            "issue": "Path traversal attack successful",
                                            "endpoint": endpoint_template,
                                            "parameter": param_name,
                                            "payload": payload,
                                            "category": category,
                                            "indicator": indicator,
                                            "severity": "critical",
                                            "response_preview": response.text[
                                                :200
                                            ],
                                        }
                                    )

                        # Test in POST body
                        try:
                            post_data = {param_name: payload, "action": "read"}
                            post_response = self.client.post(
                                endpoint_template.replace("{path}", ""),
                                json=post_data,
                            )

                            if post_response.status_code == 200:
                                post_text = post_response.text.lower()
                                for indicator in sensitive_indicators:
                                    if indicator in post_text:
                                        results["passed"] = False
                                        results["findings"].append(
                                            {
                                                "issue": "Path traversal via POST body successful",
                                                "endpoint": endpoint_template,
                                                "parameter": param_name,
                                                "payload": payload,
                                                "severity": "critical",
                                            }
                                        )

                        except Exception:
                            pass  # POST endpoint may not exist

                    except Exception as e:
                        results["findings"].append(
                            {
                                "issue": f"Error testing path traversal on {endpoint_template}",
                                "payload": payload,
                                "error": str(e),
                            }
                        )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Validate and sanitize all file path parameters",
                    "Use absolute path validation and whitelist allowed directories",
                    "Implement proper input sanitization for file operations",
                    "Never directly use user input in file system operations",
                ]
            )

        return results

    def test_static_file_serving_security(self) -> Dict[str, Any]:
        """Test static file serving security against path traversal."""
        results = {
            "test_name": "static_file_serving_security",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test common static file serving endpoints
        static_endpoints = [
            "/static/",
            "/assets/",
            "/public/",
            "/files/",
            "/uploads/",
            "/media/",
            "/images/",
            "/css/",
            "/js/",
        ]

        for endpoint in static_endpoints:
            for category, payloads in self.path_traversal_payloads.items():
                for payload in payloads[
                    :2
                ]:  # Test first 2 payloads per category
                    try:
                        # Test direct path traversal
                        test_url = f"{endpoint}{payload}"
                        response = self.client.get(test_url)

                        if response.status_code == 200:
                            # Check for sensitive file content
                            content = response.text

                            # Look for Unix passwd file format
                            if ":" in content and "root:" in content.lower():
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Static file serving allows path traversal to /etc/passwd",
                                        "endpoint": endpoint,
                                        "payload": payload,
                                        "severity": "critical",
                                        "evidence": content[:100],
                                    }
                                )

                            # Look for Windows system files
                            if (
                                "[boot loader]" in content.lower()
                                or "administrator:" in content.lower()
                            ):
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Static file serving allows path traversal to Windows system files",
                                        "endpoint": endpoint,
                                        "payload": payload,
                                        "severity": "critical",
                                    }
                                )

                            # Look for private keys
                            if (
                                "-----begin" in content.lower()
                                and "private key" in content.lower()
                            ):
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Private key accessible via path traversal",
                                        "endpoint": endpoint,
                                        "payload": payload,
                                        "severity": "critical",
                                    }
                                )

                        # Test with URL encoding
                        encoded_payload = urllib.parse.quote(payload, safe="")
                        encoded_url = f"{endpoint}{encoded_payload}"
                        encoded_response = self.client.get(encoded_url)

                        if (
                            encoded_response.status_code == 200
                            and "root:" in encoded_response.text.lower()
                        ):
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "URL-encoded path traversal successful",
                                    "endpoint": endpoint,
                                    "payload": encoded_payload,
                                    "severity": "critical",
                                }
                            )

                    except Exception as e:
                        results["findings"].append(
                            {
                                "issue": f"Error testing static file endpoint {endpoint}",
                                "payload": payload,
                                "error": str(e),
                            }
                        )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Configure web server to prevent directory traversal in static file serving",
                    "Use absolute paths and validate against allowed directories",
                    "Implement proper access controls on static file directories",
                    "Disable static file serving for non-public directories",
                ]
            )

        return results

    def test_template_inclusion_security(self) -> Dict[str, Any]:
        """Test template inclusion security against path traversal."""
        results = {
            "test_name": "template_inclusion_security",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test template-related endpoints
        template_tests = [
            ("/api/v1/render", "template"),
            ("/api/v1/include", "file"),
            ("/api/v1/view", "template_name"),
            ("/api/v1/page", "page"),
        ]

        for endpoint, param_name in template_tests:
            for category, payloads in self.path_traversal_payloads.items():
                for payload in payloads[:2]:
                    try:
                        # Test template inclusion via query parameter
                        params = {param_name: payload}
                        response = self.client.get(endpoint, params=params)

                        if response.status_code == 200:
                            content = response.text

                            # Check for file system access
                            if any(
                                indicator in content.lower()
                                for indicator in [
                                    "root:",
                                    "/bin/bash",
                                    "/bin/sh",
                                    "administrator:",
                                    "password:",
                                    "secret",
                                    "private key",
                                ]
                            ):
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Template inclusion allows file system access",
                                        "endpoint": endpoint,
                                        "parameter": param_name,
                                        "payload": payload,
                                        "severity": "high",
                                        "evidence": content[:150],
                                    }
                                )

                        # Test via POST body
                        post_data = {param_name: payload}
                        post_response = self.client.post(
                            endpoint, json=post_data
                        )

                        if post_response.status_code == 200:
                            content = post_response.text

                            if any(
                                indicator in content.lower()
                                for indicator in [
                                    "root:",
                                    "administrator:",
                                    "password:",
                                    "secret",
                                ]
                            ):
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "POST template inclusion allows file system access",
                                        "endpoint": endpoint,
                                        "parameter": param_name,
                                        "payload": payload,
                                        "severity": "high",
                                    }
                                )

                    except Exception as e:
                        results["findings"].append(
                            {
                                "issue": f"Error testing template inclusion on {endpoint}",
                                "payload": payload,
                                "error": str(e),
                            }
                        )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Validate template paths against allowed template directories",
                    "Use template sandboxing to prevent file system access",
                    "Implement whitelist of allowed template names",
                    "Never allow user input to directly specify template paths",
                ]
            )

        return results

    def test_log_file_access_prevention(self) -> Dict[str, Any]:
        """Test prevention of log file access via path traversal."""
        results = {
            "test_name": "log_file_access_prevention",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Common log file paths
        log_file_paths = [
            "/var/log/auth.log",
            "/var/log/syslog",
            "/var/log/messages",
            "/var/log/apache2/access.log",
            "/var/log/apache2/error.log",
            "/var/log/nginx/access.log",
            "/var/log/nginx/error.log",
            "/var/log/mysql/error.log",
            "/var/log/postgresql/postgresql.log",
            "C:\\Windows\\System32\\LogFiles\\W3SVC1\\ex*.log",
            "C:\\inetpub\\logs\\LogFiles\\W3SVC1\\ex*.log",
        ]

        # Test log access endpoints
        log_endpoints = [
            ("/api/v1/logs", "file"),
            ("/api/v1/admin/logs", "logfile"),
            ("/api/v1/debug/logs", "path"),
            ("/logs/", None),
        ]

        for endpoint, param_name in log_endpoints:
            for log_path in log_file_paths[:5]:  # Test first 5 log paths
                try:
                    if param_name:
                        # Test as parameter
                        params = {param_name: log_path}
                        response = self.client.get(endpoint, params=params)
                    else:
                        # Test as direct path
                        response = self.client.get(f"{endpoint}{log_path}")

                    if response.status_code == 200:
                        content = response.text.lower()

                        # Look for log file indicators
                        log_indicators = [
                            "failed password",
                            "authentication failure",
                            "invalid user",
                            "connection from",
                            "get /",
                            "post /",
                            "user-agent:",
                            "referer:",
                            "remote_addr",
                        ]

                        for indicator in log_indicators:
                            if indicator in content:
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Log file accessible via path traversal",
                                        "endpoint": endpoint,
                                        "log_path": log_path,
                                        "indicator": indicator,
                                        "severity": "high",
                                        "note": "Log files may contain sensitive information",
                                    }
                                )
                                break

                except Exception as e:
                    results["findings"].append(
                        {
                            "issue": f"Error testing log access on {endpoint}",
                            "log_path": log_path,
                            "error": str(e),
                        }
                    )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Restrict access to log files from web application",
                    "Store log files outside web-accessible directories",
                    "Implement proper access controls for log viewing functionality",
                    "Sanitize log file path parameters",
                ]
            )

        return results

    def test_configuration_file_access(self) -> Dict[str, Any]:
        """Test prevention of configuration file access."""
        results = {
            "test_name": "configuration_file_access",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test access to configuration files
        config_endpoints = [
            ("/api/v1/config", "file"),
            ("/api/v1/settings", "config"),
            ("/config/", None),
            ("/api/v1/backup", "config_file"),
        ]

        for endpoint, param_name in config_endpoints:
            for config_file in self.config_files:
                try:
                    if param_name:
                        params = {param_name: config_file}
                        response = self.client.get(endpoint, params=params)
                    else:
                        response = self.client.get(f"{endpoint}{config_file}")

                    if response.status_code == 200:
                        content = response.text.lower()

                        # Look for configuration file indicators
                        config_indicators = [
                            "password=",
                            "secret=",
                            "key=",
                            "token=",
                            "database_url",
                            "db_password",
                            "api_key",
                            "private_key",
                            "certificate",
                            "ssl_cert",
                            "redis_url",
                            "mongodb_uri",
                            "connection_string",
                        ]

                        for indicator in config_indicators:
                            if indicator in content:
                                results["passed"] = False
                                results["findings"].append(
                                    {
                                        "issue": "Configuration file accessible",
                                        "endpoint": endpoint,
                                        "config_file": config_file,
                                        "indicator": indicator,
                                        "severity": "critical",
                                        "note": "Configuration files may contain credentials",
                                    }
                                )
                                break

                except Exception as e:
                    results["findings"].append(
                        {
                            "issue": f"Error testing config access on {endpoint}",
                            "config_file": config_file,
                            "error": str(e),
                        }
                    )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Store configuration files outside web-accessible directories",
                    "Implement strict access controls for configuration endpoints",
                    "Never expose configuration files through web interface",
                    "Use environment variables instead of configuration files for secrets",
                ]
            )

        return results

    def test_directory_listing_prevention(self) -> Dict[str, Any]:
        """Test prevention of directory listing."""
        results = {
            "test_name": "directory_listing_prevention",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test common directories for listing
        test_directories = [
            "/",
            "/api/",
            "/static/",
            "/uploads/",
            "/files/",
            "/admin/",
            "/config/",
            "/logs/",
            "/backup/",
            "/tmp/",
        ]

        for directory in test_directories:
            try:
                response = self.client.get(directory)

                if response.status_code == 200:
                    content = response.text.lower()

                    # Look for directory listing indicators
                    listing_indicators = [
                        "index of",
                        "directory listing",
                        '<a href="',
                        "parent directory",
                        "[dir]",
                        "[file]",
                        "last modified",
                        "size</th>",
                        "name</th>",
                        "<pre>",
                    ]

                    listing_found = False
                    for indicator in listing_indicators:
                        if indicator in content:
                            listing_found = True
                            break

                    if listing_found:
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Directory listing enabled",
                                "directory": directory,
                                "severity": "medium",
                                "note": "Directory listing can reveal sensitive files",
                                "response_preview": response.text[:200],
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {
                        "issue": f"Error testing directory listing for {directory}",
                        "error": str(e),
                    }
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Disable directory listing in web server configuration",
                    "Ensure all directories have index files or proper access controls",
                    "Configure web server to return 403 Forbidden for directory access",
                    "Remove or secure any directories that should not be accessible",
                ]
            )

        return results

    def test_symbolic_link_exploitation(self) -> Dict[str, Any]:
        """Test prevention of symbolic link exploitation."""
        results = {
            "test_name": "symbolic_link_exploitation",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test symbolic link traversal attempts
        symlink_payloads = [
            "../../../etc/passwd",
            "symlink_to_passwd",
            "link_to_etc_passwd",
            "../symbolic_link_file",
            "soft_link_passwd",
        ]

        file_endpoints = [
            ("/api/v1/files", "filename"),
            ("/api/v1/download", "file"),
            ("/static/", None),
        ]

        for endpoint, param_name in file_endpoints:
            for payload in symlink_payloads:
                try:
                    if param_name:
                        params = {param_name: payload}
                        response = self.client.get(endpoint, params=params)
                    else:
                        response = self.client.get(f"{endpoint}{payload}")

                    if response.status_code == 200:
                        content = response.text

                        # Check if symbolic link was followed to sensitive file
                        if "root:" in content and ":0:0:" in content:
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Symbolic link exploitation possible",
                                    "endpoint": endpoint,
                                    "payload": payload,
                                    "severity": "high",
                                    "evidence": content[:100],
                                }
                            )

                except Exception as e:
                    results["findings"].append(
                        {
                            "issue": f"Error testing symbolic link on {endpoint}",
                            "payload": payload,
                            "error": str(e),
                        }
                    )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Configure application to not follow symbolic links",
                    "Validate that file paths resolve to allowed directories",
                    "Use realpath() or equivalent to resolve symbolic links",
                    "Implement checks to prevent symbolic link traversal",
                ]
            )

        return results

    def run_all_path_traversal_tests(self) -> Dict[str, Any]:
        """Run all path traversal prevention tests."""
        print("Running comprehensive path traversal prevention tests...")

        test_methods = [
            self.test_api_parameter_path_traversal,
            self.test_static_file_serving_security,
            self.test_template_inclusion_security,
            self.test_log_file_access_prevention,
            self.test_configuration_file_access,
            self.test_directory_listing_prevention,
            self.test_symbolic_link_exploitation,
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


class TestPathTraversalPrevention:
    """pytest test class for path traversal prevention validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def traversal_tester(self, client):
        """Create path traversal prevention tester."""
        return PathTraversalPreventionTester(client)

    def test_api_path_traversal_blocked(self, traversal_tester):
        """Test that API path traversal attacks are blocked."""
        result = traversal_tester.test_api_parameter_path_traversal()

        if not result["passed"]:
            critical_issues = [
                f
                for f in result.get("findings", [])
                if f.get("severity") == "critical"
            ]
            if critical_issues:
                failure_msg = (
                    "CRITICAL: API path traversal vulnerabilities detected:\n"
                )
                for finding in critical_issues:
                    failure_msg += f"  - {finding['issue']}: {finding.get('endpoint', 'unknown')}\n"
                pytest.fail(failure_msg)

    def test_static_files_secure(self, traversal_tester):
        """Test that static file serving is secure against path traversal."""
        result = traversal_tester.test_static_file_serving_security()

        if not result["passed"]:
            critical_issues = [
                f
                for f in result.get("findings", [])
                if f.get("severity") == "critical"
            ]
            if critical_issues:
                failure_msg = (
                    "CRITICAL: Static file path traversal vulnerabilities:\n"
                )
                for finding in critical_issues:
                    failure_msg += f"  - {finding['issue']}\n"
                pytest.fail(failure_msg)

    def test_config_files_protected(self, traversal_tester):
        """Test that configuration files are protected."""
        result = traversal_tester.test_configuration_file_access()

        if not result["passed"]:
            critical_issues = [
                f
                for f in result.get("findings", [])
                if f.get("severity") == "critical"
            ]
            if critical_issues:
                failure_msg = "CRITICAL: Configuration files accessible:\n"
                for finding in critical_issues:
                    failure_msg += f"  - {finding['config_file']}\n"
                pytest.fail(failure_msg)

    def test_comprehensive_path_traversal_prevention(self, traversal_tester):
        """Run comprehensive path traversal prevention tests."""
        summary = traversal_tester.run_all_path_traversal_tests()

        if summary["overall_status"] == "FAIL":
            failure_msg = f"Path traversal prevention failures: {summary['summary']['failed_tests']} out of {summary['summary']['total_tests']} tests failed\n"

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
    tester = PathTraversalPreventionTester(client)

    print("Running path traversal prevention validation tests...")
    summary = tester.run_all_path_traversal_tests()

    print(f"\n{'='*60}")
    print("PATH TRAVERSAL PREVENTION VALIDATION REPORT")
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
    report_file = f"/home/green/FreeAgentics/tests/security/path_traversal_prevention_report_{timestamp}.json"

    try:
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Exit with appropriate code
    exit(0 if summary["overall_status"] == "PASS" else 1)
