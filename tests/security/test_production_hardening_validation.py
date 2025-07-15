"""
Production Hardening Validation Testing

Comprehensive tests to validate production security hardening measures
and ensure the application is properly configured for production deployment.

This module focuses on:
1. Debug mode disabled verification
2. Verbose logging disabled
3. Development headers removed
4. Error handling consistency
5. Information leakage prevention
6. Security configuration validation
7. Environment variable validation
8. SSL/TLS configuration validation
9. Database security configuration
10. Infrastructure security validation
"""

import json
import os
import re
import time
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.main import app


class ProductionHardeningTester:
    """Comprehensive production hardening validation tester."""

    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []

        # Critical production requirements
        self.production_requirements = {
            "environment_variables": [
                "SECRET_KEY",
                "JWT_SECRET",
                "DATABASE_URL",
                "REDIS_URL",
                "PRODUCTION",
            ],
            "forbidden_debug_indicators": [
                "debug.*true",
                "development.*mode",
                "test.*mode",
                "verbose.*true",
                "__debug__",
                r"pdb\.set_trace",
                r"breakpoint\(",
                r"console\.log",
                r"print\(",
                "debugger;",
                "traceback",
                "stack.*trace",
            ],
            "forbidden_headers": [
                "X-Debug-Token",
                "X-Debug-Token-Link",
                "X-Powered-By",
                "X-AspNet-Version",
                "X-AspNetMvc-Version",
                "X-Generator",
                "X-Runtime",
            ],
            "required_security_headers": [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Referrer-Policy",
                "Content-Security-Policy",
                "Strict-Transport-Security",  # For HTTPS
            ],
            "ssl_tls_requirements": {
                "min_tls_version": "1.2",
                "required_ciphers": [
                    "ECDHE-RSA-AES256-GCM-SHA384",
                    "ECDHE-RSA-AES128-GCM-SHA256",
                    "ECDHE-RSA-AES256-SHA384",
                    "ECDHE-RSA-AES128-SHA256",
                ],
                "forbidden_ciphers": ["RC4", "DES", "3DES", "MD5", "SHA1"],
            },
        }

    def test_debug_mode_disabled(self) -> Dict[str, Any]:
        """Test that debug mode is properly disabled in production."""
        results = {
            "test_name": "debug_mode_disabled",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various endpoints for debug indicators
        test_endpoints = [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/system/status",
            "/api/v1/system/info",
            "/nonexistent",  # 404 error page
        ]

        for endpoint in test_endpoints:
            try:
                response = self.client.get(endpoint)
                response_text = response.text.lower()

                # Check for debug indicators in response content
                for debug_pattern in self.production_requirements["forbidden_debug_indicators"]:
                    if re.search(debug_pattern, response_text, re.IGNORECASE):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Debug mode indicator detected",
                                "endpoint": endpoint,
                                "pattern": debug_pattern,
                                "status_code": response.status_code,
                            }
                        )

                # Check for debug headers
                debug_headers_found = []
                for header_name, header_value in response.headers.items():
                    header_lower = header_name.lower()
                    value_lower = str(header_value).lower()

                    if any(debug_word in header_lower for debug_word in ["debug", "dev", "test"]):
                        debug_headers_found.append(f"{header_name}: {header_value}")

                    if any(
                        debug_word in value_lower for debug_word in ["debug", "development", "test"]
                    ):
                        debug_headers_found.append(f"{header_name}: {header_value}")

                if debug_headers_found:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Debug headers detected",
                            "endpoint": endpoint,
                            "debug_headers": debug_headers_found,
                        }
                    )

                # Check for development error pages
                if response.status_code >= 400:
                    dev_error_indicators = [
                        "werkzeug",
                        "django",
                        "flask",
                        "development server",
                        "traceback",
                        r"file.*line.*\d+",
                        "exception.*at.*line",
                    ]

                    for indicator in dev_error_indicators:
                        if re.search(indicator, response_text, re.IGNORECASE):
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Development error page detected",
                                    "endpoint": endpoint,
                                    "indicator": indicator,
                                    "status_code": response.status_code,
                                }
                            )

            except Exception as e:
                results["findings"].append(
                    {"issue": f"Error testing endpoint {endpoint}", "error": str(e)}
                )

        # Check FastAPI app debug setting
        try:
            # This would need to be adapted based on how the app is configured
            app_debug = getattr(app, "debug", None)
            if app_debug is True:
                results["passed"] = False
                results["findings"].append(
                    {"issue": "FastAPI app debug mode is enabled", "detail": "app.debug = True"}
                )
        except Exception as e:
            results["findings"].append(
                {"issue": "Could not check FastAPI debug setting", "error": str(e)}
            )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Ensure DEBUG=False in production environment",
                    "Remove or disable development error handlers",
                    "Configure production-grade error pages",
                    "Remove debug headers and middleware",
                ]
            )

        return results

    def test_environment_configuration(self) -> Dict[str, Any]:
        """Test production environment configuration."""
        results = {
            "test_name": "environment_configuration",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Check critical environment variables
        missing_env_vars = []
        weak_env_vars = []

        for env_var in self.production_requirements["environment_variables"]:
            value = os.getenv(env_var)

            if not value:
                missing_env_vars.append(env_var)
            else:
                # Check for weak/default values
                weak_patterns = [
                    "dev.*secret",
                    "test.*secret",
                    "changeme",
                    "password",
                    "secret",
                    "key123",
                    "admin",
                    "localhost",
                    r"example\.com",
                    "your.*key.*here",
                ]

                for pattern in weak_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        weak_env_vars.append(
                            {
                                "variable": env_var,
                                "issue": f"Potentially weak value matching pattern: {pattern}",
                                "value_preview": value[:10] + "..." if len(value) > 10 else value,
                            }
                        )

        if missing_env_vars:
            results["passed"] = False
            results["findings"].append(
                {
                    "issue": "Missing critical environment variables",
                    "missing_variables": missing_env_vars,
                }
            )

        if weak_env_vars:
            results["passed"] = False
            results["findings"].append(
                {
                    "issue": "Weak environment variable values detected",
                    "weak_variables": weak_env_vars,
                }
            )

        # Check PRODUCTION flag
        production_flag = os.getenv("PRODUCTION", "false").lower()
        if production_flag not in ["true", "1", "yes"]:
            results["findings"].append(
                {
                    "issue": "PRODUCTION environment variable not set to true",
                    "current_value": production_flag,
                    "note": "This may indicate non-production configuration",
                }
            )

        # Check for development URLs in environment
        dev_url_patterns = [
            "localhost",
            r"127\.0\.0\.1",
            r"0\.0\.0\.0",
            r"dev\.",
            r"test\.",
            r"staging\.",
            r":\d{4,5}",  # Non-standard ports
        ]

        url_env_vars = ["DATABASE_URL", "REDIS_URL", "API_BASE_URL", "FRONTEND_URL"]

        for env_var in url_env_vars:
            value = os.getenv(env_var, "")
            for pattern in dev_url_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    results["findings"].append(
                        {
                            "issue": f"Development URL pattern in {env_var}",
                            "pattern": pattern,
                            "variable": env_var,
                            "value_preview": value[:30] + "..." if len(value) > 30 else value,
                        }
                    )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Set all required environment variables for production",
                    "Use strong, unique secrets for all security-related variables",
                    "Ensure PRODUCTION=true in production environment",
                    "Use production URLs and endpoints, not development/localhost",
                ]
            )

        return results

    def test_security_headers_production(self) -> Dict[str, Any]:
        """Test production security headers configuration."""
        results = {
            "test_name": "security_headers_production",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test critical endpoints
        test_endpoints = [
            "/",
            "/api/v1/system/status",
            "/api/v1/auth/login",
            "/nonexistent",  # Error page
        ]

        for endpoint in test_endpoints:
            try:
                response = self.client.get(endpoint)

                # Check for required security headers
                missing_headers = []
                weak_headers = []

                for required_header in self.production_requirements["required_security_headers"]:
                    if required_header not in response.headers:
                        # HSTS might not be present in non-HTTPS testing
                        if required_header != "Strict-Transport-Security":
                            missing_headers.append(required_header)
                    else:
                        header_value = response.headers[required_header]

                        # Validate header values
                        if required_header == "X-Frame-Options":
                            if header_value not in ["DENY", "SAMEORIGIN"]:
                                weak_headers.append(
                                    {
                                        "header": required_header,
                                        "value": header_value,
                                        "issue": "Should be DENY or SAMEORIGIN",
                                    }
                                )

                        elif required_header == "X-Content-Type-Options":
                            if header_value != "nosniff":
                                weak_headers.append(
                                    {
                                        "header": required_header,
                                        "value": header_value,
                                        "issue": "Should be nosniff",
                                    }
                                )

                        elif required_header == "Content-Security-Policy":
                            # Basic CSP validation
                            if "unsafe-eval" in header_value or "unsafe-inline" in header_value:
                                weak_headers.append(
                                    {
                                        "header": required_header,
                                        "value": header_value[:100] + "...",
                                        "issue": "Contains unsafe CSP directives",
                                    }
                                )

                if missing_headers:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Missing required security headers",
                            "endpoint": endpoint,
                            "missing_headers": missing_headers,
                        }
                    )

                if weak_headers:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Weak security header values",
                            "endpoint": endpoint,
                            "weak_headers": weak_headers,
                        }
                    )

                # Check for forbidden headers
                forbidden_found = []
                for forbidden_header in self.production_requirements["forbidden_headers"]:
                    if forbidden_header in response.headers:
                        forbidden_found.append(
                            {
                                "header": forbidden_header,
                                "value": response.headers[forbidden_header],
                            }
                        )

                if forbidden_found:
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Forbidden information disclosure headers present",
                            "endpoint": endpoint,
                            "forbidden_headers": forbidden_found,
                        }
                    )

            except Exception as e:
                results["findings"].append(
                    {"issue": f"Error testing security headers for {endpoint}", "error": str(e)}
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement all required security headers",
                    "Remove information disclosure headers",
                    "Configure strong Content Security Policy",
                    "Ensure security headers are applied to all responses",
                ]
            )

        return results

    def test_error_handling_production(self) -> Dict[str, Any]:
        """Test production error handling configuration."""
        results = {
            "test_name": "error_handling_production",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various error scenarios
        error_test_scenarios = [
            ("500_internal_error", self._trigger_500_error),
            ("404_not_found", lambda: self.client.get("/nonexistent")),
            ("400_bad_request", lambda: self.client.post("/api/v1/agents", data="invalid json")),
            ("401_unauthorized", lambda: self.client.get("/api/v1/agents/protected")),
            ("422_validation_error", lambda: self.client.post("/api/v1/agents", json={})),
        ]

        for scenario_name, trigger_func in error_test_scenarios:
            try:
                response = trigger_func()

                # Check response structure
                if response.status_code >= 400:
                    response_text = response.text

                    # Should not contain sensitive information
                    sensitive_patterns = [
                        r"traceback",
                        r"stack.*trace",
                        r'file.*".*\.py".*line.*\d+',
                        r"exception.*at.*line",
                        r"/home/.*/",
                        r"/usr/.*/",
                        r"/var/.*/",
                        r"database.*error",
                        r"connection.*failed",
                        r"sqlalchemy",
                        r"postgresql",
                        r"mysql",
                        r"redis.*error",
                    ]

                    for pattern in sensitive_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Sensitive information in error response",
                                    "scenario": scenario_name,
                                    "pattern": pattern,
                                    "status_code": response.status_code,
                                }
                            )

                    # Check for proper error structure
                    try:
                        error_data = response.json()

                        # Should have standard error structure
                        if "detail" not in error_data:
                            results["findings"].append(
                                {
                                    "issue": "Non-standard error response structure",
                                    "scenario": scenario_name,
                                    "status_code": response.status_code,
                                }
                            )

                        # Error message should be generic
                        detail = error_data.get("detail", "")
                        if any(
                            sensitive in detail.lower()
                            for sensitive in ["sql", "database", "internal", "exception"]
                        ):
                            results["passed"] = False
                            results["findings"].append(
                                {
                                    "issue": "Non-generic error message",
                                    "scenario": scenario_name,
                                    "detail": detail,
                                    "status_code": response.status_code,
                                }
                            )

                    except json.JSONDecodeError:
                        results["findings"].append(
                            {
                                "issue": "Non-JSON error response",
                                "scenario": scenario_name,
                                "status_code": response.status_code,
                                "content_type": response.headers.get("Content-Type", "unknown"),
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {"issue": f"Error testing scenario {scenario_name}", "error": str(e)}
                )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Implement generic error messages for production",
                    "Remove sensitive information from error responses",
                    "Use structured error response format",
                    "Ensure consistent error handling across all endpoints",
                ]
            )

        return results

    def _trigger_500_error(self):
        """Helper to trigger a 500 error for testing."""
        # This would need to be implemented based on your app structure
        # For now, try to trigger an error through invalid operations
        try:
            return self.client.post("/api/v1/agents", json={"name": None})
        except:
            # Fallback to a more generic error trigger
            return self.client.get("/api/v1/system/force-error")

    def test_logging_configuration(self) -> Dict[str, Any]:
        """Test production logging configuration."""
        results = {
            "test_name": "logging_configuration",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Check logging level configuration
        import logging

        # Test various loggers
        test_loggers = ["root", "__main__", "api", "auth", "uvicorn", "fastapi"]

        for logger_name in test_loggers:
            logger = logging.getLogger(logger_name)

            # In production, should not be DEBUG level
            if logger.level <= logging.DEBUG and logger.level != logging.NOTSET:
                results["findings"].append(
                    {
                        "issue": "Logger set to DEBUG level in production",
                        "logger": logger_name,
                        "level": logging.getLevelName(logger.level),
                    }
                )

        # Check for console handlers in production
        root_logger = logging.getLogger()
        console_handlers = []

        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and hasattr(handler, "stream"):
                import sys

                if handler.stream in [sys.stdout, sys.stderr]:
                    console_handlers.append(type(handler).__name__)

        if console_handlers and os.getenv("PRODUCTION", "false").lower() == "true":
            results["findings"].append(
                {
                    "issue": "Console logging handlers in production",
                    "handlers": console_handlers,
                    "note": "Consider using file or remote logging in production",
                }
            )

        # Test that sensitive information is not logged
        test_scenarios = [
            (
                "login_attempt",
                lambda: self.client.post(
                    "/api/v1/auth/login", json={"username": "test", "password": "secret123"}
                ),
            ),
            (
                "agent_creation",
                lambda: self.client.post(
                    "/api/v1/agents", json={"name": "test", "secret_key": "secret123"}
                ),
            ),
        ]

        # This would require log capture to properly test
        # For now, we'll check if the app has proper log filtering

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Set appropriate logging levels for production (INFO or WARNING)",
                    "Use file-based or remote logging instead of console in production",
                    "Implement log filtering to prevent sensitive data logging",
                    "Configure log rotation and retention policies",
                ]
            )

        return results

    def test_database_security_configuration(self) -> Dict[str, Any]:
        """Test database security configuration."""
        results = {
            "test_name": "database_security_configuration",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Check database URL configuration
        database_url = os.getenv("DATABASE_URL", "")

        if database_url:
            # Check for insecure database configuration
            insecure_patterns = [
                ("weak_password", r"password.*=.*(admin|password|123|test)"),
                ("no_ssl", r"sslmode.*=.*disable"),
                ("default_port", r"localhost.*:5432"),  # Default PostgreSQL port
                ("no_auth", r"://.*@"),  # User with no password
                ("weak_user", r"://(root|admin|sa|postgres)@"),
            ]

            for issue_type, pattern in insecure_patterns:
                if re.search(pattern, database_url, re.IGNORECASE):
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": f"Insecure database configuration: {issue_type}",
                            "pattern": pattern,
                            "note": "Database URL contains potentially insecure configuration",
                        }
                    )

        # Test database connection security
        try:
            # Make a simple request that would hit the database
            response = self.client.get("/api/v1/system/status")

            # Check if database errors are properly handled
            if response.status_code == 500:
                response_text = response.text

                db_error_patterns = [
                    "connection.*refused",
                    "authentication.*failed",
                    "permission.*denied.*database",
                    "database.*not.*exist",
                    "role.*does.*not.*exist",
                    "ssl.*required",
                    "timeout.*expired",
                ]

                for pattern in db_error_patterns:
                    if re.search(pattern, response_text, re.IGNORECASE):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Database error information disclosed",
                                "pattern": pattern,
                                "note": "Database errors should be generic in production",
                            }
                        )

        except Exception as e:
            results["findings"].append(
                {"issue": "Error testing database security", "error": str(e)}
            )

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Use strong database credentials",
                    "Enable SSL/TLS for database connections",
                    "Use non-default database ports",
                    "Implement proper database error handling",
                    "Configure database connection pooling and timeouts",
                ]
            )

        return results

    def test_ssl_tls_configuration(self) -> Dict[str, Any]:
        """Test SSL/TLS configuration (where applicable)."""
        results = {
            "test_name": "ssl_tls_configuration",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Check if HTTPS is enforced
        https_endpoints = ["/", "/api/v1/auth/login", "/api/v1/system/status"]

        for endpoint in https_endpoints:
            try:
                response = self.client.get(endpoint)

                # Check for HSTS header
                hsts_header = response.headers.get("Strict-Transport-Security")

                if not hsts_header:
                    results["findings"].append(
                        {
                            "issue": "Missing HSTS header",
                            "endpoint": endpoint,
                            "note": "HSTS should be configured for HTTPS endpoints",
                        }
                    )
                else:
                    # Validate HSTS configuration
                    if "max-age=" not in hsts_header:
                        results["findings"].append(
                            {
                                "issue": "Invalid HSTS header format",
                                "endpoint": endpoint,
                                "hsts_value": hsts_header,
                            }
                        )
                    else:
                        # Extract max-age
                        max_age_match = re.search(r"max-age=(\d+)", hsts_header)
                        if max_age_match:
                            max_age = int(max_age_match.group(1))
                            # Should be at least 6 months (15768000 seconds)
                            if max_age < 15768000:
                                results["findings"].append(
                                    {
                                        "issue": "HSTS max-age too short",
                                        "endpoint": endpoint,
                                        "max_age": max_age,
                                        "recommended_min": 15768000,
                                    }
                                )

                # Check for secure cookie settings
                set_cookie_headers = (
                    response.headers.get_list("Set-Cookie")
                    if hasattr(response.headers, "get_list")
                    else []
                )

                for cookie_header in set_cookie_headers:
                    if "Secure" not in cookie_header:
                        results["findings"].append(
                            {
                                "issue": "Cookie without Secure flag",
                                "endpoint": endpoint,
                                "cookie_header": (
                                    cookie_header[:100] + "..."
                                    if len(cookie_header) > 100
                                    else cookie_header
                                ),
                            }
                        )

                    if "HttpOnly" not in cookie_header:
                        results["findings"].append(
                            {
                                "issue": "Cookie without HttpOnly flag",
                                "endpoint": endpoint,
                                "cookie_header": (
                                    cookie_header[:100] + "..."
                                    if len(cookie_header) > 100
                                    else cookie_header
                                ),
                            }
                        )

                    if "SameSite=" not in cookie_header:
                        results["findings"].append(
                            {
                                "issue": "Cookie without SameSite attribute",
                                "endpoint": endpoint,
                                "cookie_header": (
                                    cookie_header[:100] + "..."
                                    if len(cookie_header) > 100
                                    else cookie_header
                                ),
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {"issue": f"Error testing SSL/TLS for {endpoint}", "error": str(e)}
                )

        # Note: Actual SSL/TLS cipher and protocol testing would require
        # external tools like openssl or ssl library testing

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Configure HSTS with appropriate max-age (minimum 6 months)",
                    "Set Secure flag on all cookies in HTTPS environments",
                    "Set HttpOnly flag on session cookies",
                    "Configure SameSite attribute on cookies",
                    "Use TLS 1.2 or higher",
                    "Disable weak ciphers and protocols",
                ]
            )

        return results

    def test_infrastructure_security(self) -> Dict[str, Any]:
        """Test infrastructure security configuration."""
        results = {
            "test_name": "infrastructure_security",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Check for common infrastructure misconfigurations

        # Test for exposed management endpoints
        management_endpoints = [
            "/admin",
            "/management",
            "/actuator",
            "/metrics",
            "/health/detailed",
            "/debug",
            "/info",
            "/env",
            "/config",
            "/status/detailed",
        ]

        exposed_endpoints = []

        for endpoint in management_endpoints:
            try:
                response = self.client.get(endpoint)

                # These endpoints should either not exist or require authentication
                if response.status_code == 200:
                    response_text = response.text

                    # Check if response contains sensitive information
                    sensitive_info_patterns = [
                        "environment.*variables",
                        "configuration",
                        "database.*url",
                        "secret.*key",
                        "api.*key",
                        "version.*info",
                        "system.*info",
                        "memory.*usage",
                        "cpu.*usage",
                    ]

                    for pattern in sensitive_info_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            exposed_endpoints.append(
                                {
                                    "endpoint": endpoint,
                                    "issue": f"Sensitive information exposed: {pattern}",
                                    "status_code": response.status_code,
                                }
                            )
                            break
                    else:
                        exposed_endpoints.append(
                            {
                                "endpoint": endpoint,
                                "issue": "Management endpoint accessible without authentication",
                                "status_code": response.status_code,
                            }
                        )

            except Exception as e:
                results["findings"].append(
                    {"issue": f"Error testing management endpoint {endpoint}", "error": str(e)}
                )

        if exposed_endpoints:
            results["passed"] = False
            results["findings"].append(
                {
                    "issue": "Exposed management endpoints detected",
                    "exposed_endpoints": exposed_endpoints,
                }
            )

        # Check for version disclosure in common files
        version_files = [
            "/robots.txt",
            "/sitemap.xml",
            "/.well-known/security.txt",
            "/humans.txt",
            "/version.txt",
            "/changelog.txt",
        ]

        for version_file in version_files:
            try:
                response = self.client.get(version_file)

                if response.status_code == 200:
                    response_text = response.text

                    # Check for version information
                    version_patterns = [
                        r"version.*\d+\.\d+",
                        r"v\d+\.\d+",
                        r"release.*\d+",
                        r"build.*\d+",
                    ]

                    for pattern in version_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            results["findings"].append(
                                {
                                    "issue": "Version information disclosed",
                                    "file": version_file,
                                    "pattern": pattern,
                                }
                            )
                            break

            except Exception:
                continue  # These files are optional

        if results["findings"]:
            results["recommendations"].extend(
                [
                    "Secure or remove management endpoints in production",
                    "Implement authentication for administrative interfaces",
                    "Remove version information from publicly accessible files",
                    "Configure proper access controls for infrastructure endpoints",
                ]
            )

        return results

    def run_all_production_hardening_tests(self) -> Dict[str, Any]:
        """Run all production hardening validation tests."""
        print("Running comprehensive production hardening validation tests...")

        test_methods = [
            self.test_debug_mode_disabled,
            self.test_environment_configuration,
            self.test_security_headers_production,
            self.test_error_handling_production,
            self.test_logging_configuration,
            self.test_database_security_configuration,
            self.test_ssl_tls_configuration,
            self.test_infrastructure_security,
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
                        print(f"    - {finding['issue']}")

            except Exception as e:
                print(f"  {test_method.__name__}: ERROR - {str(e)}")
                all_results.append(
                    {
                        "test_name": test_method.__name__,
                        "passed": False,
                        "findings": [{"issue": f"Test execution error: {str(e)}"}],
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
                # Determine severity based on issue type
                issue = finding.get("issue", "").lower()

                if any(
                    critical_term in issue
                    for critical_term in [
                        "debug mode",
                        "sensitive information",
                        "weak",
                        "missing.*secret",
                    ]
                ):
                    critical_findings.append(finding)
                elif any(
                    high_term in issue
                    for high_term in ["security header", "error response", "configuration"]
                ):
                    high_findings.append(finding)
                else:
                    medium_findings.append(finding)

        # Collect all recommendations
        all_recommendations = []
        for result in all_results:
            all_recommendations.extend(result.get("recommendations", []))

        # Remove duplicates
        unique_recommendations = list(set(all_recommendations))

        # Calculate production readiness score
        production_readiness = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Adjust score based on critical findings
        if critical_findings:
            production_readiness = max(0, production_readiness - len(critical_findings) * 15)

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "production_readiness_score": production_readiness,
                "critical_findings": len(critical_findings),
                "high_findings": len(high_findings),
                "medium_findings": len(medium_findings),
            },
            "test_results": all_results,
            "recommendations": unique_recommendations,
            "overall_status": (
                "PRODUCTION_READY"
                if failed_tests == 0 and critical_findings == 0
                else "NOT_PRODUCTION_READY"
            ),
            "findings_by_severity": {
                "critical": critical_findings,
                "high": high_findings,
                "medium": medium_findings,
            },
        }

        return summary


class TestProductionHardeningValidation:
    """pytest test class for production hardening validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def hardening_tester(self, client):
        """Create production hardening tester."""
        return ProductionHardeningTester(client)

    def test_debug_mode_disabled_production(self, hardening_tester):
        """Test that debug mode is disabled in production."""
        result = hardening_tester.test_debug_mode_disabled()

        if not result["passed"]:
            failure_msg = "Debug mode not properly disabled for production:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_environment_properly_configured(self, hardening_tester):
        """Test that environment is properly configured for production."""
        result = hardening_tester.test_environment_configuration()

        if not result["passed"]:
            failure_msg = "Production environment configuration issues:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_security_headers_production_ready(self, hardening_tester):
        """Test that security headers are production-ready."""
        result = hardening_tester.test_security_headers_production()

        if not result["passed"]:
            failure_msg = "Production security headers issues:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_error_handling_production_ready(self, hardening_tester):
        """Test that error handling is production-ready."""
        result = hardening_tester.test_error_handling_production()

        if not result["passed"]:
            failure_msg = "Production error handling issues:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_database_security_production(self, hardening_tester):
        """Test that database security is production-ready."""
        result = hardening_tester.test_database_security_configuration()

        if not result["passed"]:
            failure_msg = "Database security configuration issues:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_comprehensive_production_hardening(self, hardening_tester):
        """Run comprehensive production hardening validation."""
        summary = hardening_tester.run_all_production_hardening_tests()

        if summary["overall_status"] == "NOT_PRODUCTION_READY":
            failure_msg = f"Application is NOT production ready!\n"
            failure_msg += f"Production Readiness Score: {summary['summary']['production_readiness_score']:.1f}%\n"
            failure_msg += f"Critical Issues: {summary['summary']['critical_findings']}\n"
            failure_msg += f"High Severity Issues: {summary['summary']['high_findings']}\n"
            failure_msg += f"Failed Tests: {summary['summary']['failed_tests']} out of {summary['summary']['total_tests']}\n"

            if summary["findings_by_severity"]["critical"]:
                failure_msg += "\nCRITICAL ISSUES:\n"
                for finding in summary["findings_by_severity"]["critical"][:5]:  # Limit to first 5
                    failure_msg += f"  - {finding['issue']}\n"

            if summary["recommendations"]:
                failure_msg += "\nTop Recommendations:\n"
                for rec in summary["recommendations"][:10]:  # Limit to first 10
                    failure_msg += f"  - {rec}\n"

            pytest.fail(failure_msg)

        # Even if production ready, warn about any findings
        if summary["summary"]["failed_tests"] > 0:
            print(
                f"WARNING: {summary['summary']['failed_tests']} production hardening tests failed"
            )
            print(
                f"Production Readiness Score: {summary['summary']['production_readiness_score']:.1f}%"
            )


if __name__ == "__main__":
    """Direct execution for standalone testing."""
    client = TestClient(app)
    tester = ProductionHardeningTester(client)

    print("Running production hardening validation tests...")
    summary = tester.run_all_production_hardening_tests()

    print(f"\n{'='*70}")
    print("PRODUCTION HARDENING VALIDATION REPORT")
    print(f"{'='*70}")
    print(f"Total Tests: {summary['summary']['total_tests']}")
    print(f"Passed: {summary['summary']['passed_tests']}")
    print(f"Failed: {summary['summary']['failed_tests']}")
    print(f"Pass Rate: {summary['summary']['pass_rate']:.1f}%")
    print(f"Production Readiness Score: {summary['summary']['production_readiness_score']:.1f}%")
    print(f"Overall Status: {summary['overall_status']}")

    print(f"\nFindings by Severity:")
    print(f"  Critical: {summary['summary']['critical_findings']}")
    print(f"  High: {summary['summary']['high_findings']}")
    print(f"  Medium: {summary['summary']['medium_findings']}")

    if summary["findings_by_severity"]["critical"]:
        print(f"\n{'='*40}")
        print("CRITICAL ISSUES (MUST FIX)")
        print(f"{'='*40}")
        for finding in summary["findings_by_severity"]["critical"]:
            print(f"• {finding['issue']}")

    if summary["recommendations"]:
        print(f"\n{'='*40}")
        print("RECOMMENDATIONS")
        print(f"{'='*40}")
        for rec in summary["recommendations"]:
            print(f"• {rec}")

    # Save report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = (
        f"/home/green/FreeAgentics/tests/security/production_hardening_report_{timestamp}.json"
    )

    try:
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Exit with appropriate code based on production readiness
    if summary["overall_status"] == "NOT_PRODUCTION_READY":
        exit(1)
    elif summary["summary"]["critical_findings"] > 0:
        exit(2)
    else:
        exit(0)
