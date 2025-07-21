"""
Comprehensive Error Handling Information Disclosure Testing for FreeAgentics Platform

This module tests for security vulnerabilities related to error handling and information disclosure,
including database error leakage, stack trace exposure, debug information disclosure,
internal path revelation, and version information leakage.

Security Tests Coverage:
1. Database Error Information Leakage
2. Stack Trace Exposure Detection
3. Debug Information Disclosure
4. Internal Path Revelation
5. Version Information Leakage
6. Exception Handling Testing
7. Error Response Standardization
8. HTTP Status Code Consistency
9. Error Message Sanitization
10. Production Hardening Validation
"""

import json
import logging
import re
import time
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api.main import app


class ErrorDisclosureTestCase:
    """Base class for error disclosure test cases."""

    def __init__(self, name: str, description: str, category: str):
        self.name = name
        self.description = description
        self.category = category
        self.passed = False
        self.error_details = None
        self.response_data = None


class ErrorHandlingTester:
    """Comprehensive error handling and information disclosure tester."""

    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []
        self.logger = logging.getLogger(__name__)

        # Common patterns that should NOT be exposed in production
        self.sensitive_patterns = {
            "database_errors": [
                r"ORA-\d+",  # Oracle errors
                r"SQL Server.*Error",  # SQL Server errors
                r"MySQLSyntaxErrorException",  # MySQL errors
                r"PostgreSQL.*Error",  # PostgreSQL errors
                r"SQLAlchemyError",  # SQLAlchemy errors
                r"IntegrityError",  # Database integrity errors
                r"OperationalError",  # Database operational errors
                r"ProgrammingError",  # Database programming errors
                r"DataError",  # Database data errors
                r"InternalError",  # Database internal errors
                r"NotSupportedError",  # Database not supported errors
                r"InvalidRequestError",  # SQLAlchemy invalid request errors
                r"StatementError",  # SQLAlchemy statement errors
                r"CompileError",  # SQLAlchemy compile errors
                r"DisconnectionError",  # Database disconnection errors
                r"TimeoutError",  # Database timeout errors
                r"DatabaseError",  # Generic database errors
                r"connection.*failed",  # Connection failures
                r"authentication.*failed",  # DB auth failures
                r"table.*does not exist",  # Table existence errors
                r"column.*does not exist",  # Column existence errors
                r"foreign key constraint",  # FK constraint errors
                r"unique constraint",  # Unique constraint errors
                r"check constraint",  # Check constraint errors
                r"permission denied.*table",  # Table permission errors
                r"relation.*does not exist",  # PostgreSQL relation errors
                r"access denied.*database",  # Database access errors
                r"deadlock detected",  # Deadlock errors
                r"lock timeout exceeded",  # Lock timeout errors
                r"out of memory",  # Memory errors
                r"disk full",  # Disk space errors
                r"too many connections",  # Connection limit errors
                r"server has gone away",  # MySQL server gone away
                r"connection.*reset",  # Connection reset errors
                r"connection.*timed out",  # Connection timeout errors
                r"host.*not found",  # Host resolution errors
                r"network.*unreachable",  # Network errors
                r"connection.*refused",  # Connection refused errors
            ],
            "stack_traces": [
                r"Traceback \(most recent call last\)",  # Python stack traces
                r'File ".*", line \d+',  # Python file line references
                r"at .*\(.*:\d+:\d+\)",  # JavaScript stack traces
                r"at .*\..*\(.*\.py:\d+\)",  # Python method stack traces
                r"raise.*Error",  # Python exception raising
                r".*Exception.*:",  # Exception class references
                r".*Error.*:",  # Error class references
                r"StackTrace:",  # Explicit stack trace headers
                r"CallStack:",  # Call stack references
                r"^\s*at\s+",  # Stack frame indicators
                r"^\s*in\s+",  # Python 'in' indicators
                r"^\s*File\s+",  # Python file indicators
                r"line\s+\d+",  # Line number references
                r"module\s+.*\.py",  # Python module references
                r"function\s+.*\(",  # Function call references
                r"method\s+.*\.",  # Method call references
                r"class\s+.*\.",  # Class references in traces
                r"def\s+.*\(",  # Function definition references
                r"async\s+def\s+.*\(",  # Async function references
                r"__.*__\(",  # Python special method references
                r"self\..*\(",  # Instance method references
                r"cls\..*\(",  # Class method references
            ],
            "debug_info": [
                r"DEBUG:",  # Debug log levels
                r"TRACE:",  # Trace log levels
                r"<.*object at 0x.*>",  # Python object representations
                r"memory.*0x[0-9a-fA-F]+",  # Memory addresses
                r"id=\d+",  # Object ID references
                r"__debug__",  # Python debug flag
                r"pdb\.set_trace",  # Python debugger
                r"breakpoint\(",  # Python breakpoint calls
                r"print\(.*\)",  # Print statements (debug)
                r"console\.log",  # JavaScript console logs
                r"console\.debug",  # JavaScript debug logs
                r"console\.trace",  # JavaScript trace logs
                r"console\.error",  # JavaScript error logs
                r"console\.warn",  # JavaScript warning logs
                r"console\.info",  # JavaScript info logs
                r"debugger;",  # JavaScript debugger statements
                r"development.*mode",  # Development mode indicators
                r"debug.*mode",  # Debug mode indicators
                r"test.*mode",  # Test mode indicators
                r"verbose.*mode",  # Verbose mode indicators
                r"profiler.*",  # Code profiler references
                r"benchmark.*",  # Benchmark references
                r"performance.*timing",  # Performance timing info
                r"execution.*time",  # Execution time info
                r"query.*time",  # Database query timing
                r"response.*time",  # Response time info
                r"memory.*usage",  # Memory usage info
                r"cpu.*usage",  # CPU usage info
                r"disk.*usage",  # Disk usage info
                r"cache.*miss",  # Cache miss info
                r"cache.*hit",  # Cache hit info
            ],
            "internal_paths": [
                r"/home/.*/",  # Unix home directories
                r"/usr/.*/",  # Unix system directories
                r"/var/.*/",  # Unix variable directories
                r"/opt/.*/",  # Unix optional directories
                r"/tmp/.*/",  # Unix temporary directories
                r"/etc/.*/",  # Unix configuration directories
                r"C:\\.*\\",  # Windows paths
                r"D:\\.*\\",  # Windows paths (other drives)
                r"\\\\.*\\",  # Windows UNC paths
                r"/app/.*/",  # Docker app directories
                r"/code/.*/",  # Common code directories
                r"/src/.*/",  # Source directories
                r"/project/.*/",  # Project directories
                r"/workspace/.*/",  # Workspace directories
                r"/repository/.*/",  # Repository directories
                r"/git/.*/",  # Git directories
                r"/deploy/.*/",  # Deployment directories
                r"/build/.*/",  # Build directories
                r"/dist/.*/",  # Distribution directories
                r"/node_modules/.*/",  # Node.js modules
                r"/venv/.*/",  # Python virtual environments
                r"/env/.*/",  # Environment directories
                r"\.pyc",  # Python compiled files
                r"\.pyo",  # Python optimized files
                r"\.pyd",  # Python extension files
                r"\.so",  # Shared object files
                r"\.dll",  # Windows DLL files
                r"\.exe",  # Windows executable files
                r"\.class",  # Java class files
                r"\.jar",  # Java archive files
                r"\.war",  # Java web archive files
                r"/green/FreeAgentics",  # Specific project paths
                r"/home/green/",  # Specific user paths
            ],
            "version_info": [
                r"Python/[\d\.]+",  # Python version
                r"FastAPI/[\d\.]+",  # FastAPI version
                r"SQLAlchemy/[\d\.]+",  # SQLAlchemy version
                r"Pydantic/[\d\.]+",  # Pydantic version
                r"uvicorn/[\d\.]+",  # Uvicorn version
                r"nginx/[\d\.]+",  # Nginx version
                r"Apache/[\d\.]+",  # Apache version
                r"OpenSSL/[\d\.]+",  # OpenSSL version
                r"PostgreSQL/[\d\.]+",  # PostgreSQL version
                r"MySQL/[\d\.]+",  # MySQL version
                r"Redis/[\d\.]+",  # Redis version
                r"Node\.js/[\d\.]+",  # Node.js version
                r"npm/[\d\.]+",  # npm version
                r"yarn/[\d\.]+",  # Yarn version
                r"Docker/[\d\.]+",  # Docker version
                r"Kubernetes/[\d\.]+",  # Kubernetes version
                r"version.*[\d\.]+",  # Generic version patterns
                r"v[\d\.]+",  # Version prefixed with 'v'
                r"release.*[\d\.]+",  # Release version patterns
                r"build.*[\d\.]+",  # Build version patterns
                r"commit.*[0-9a-fA-F]{7,}",  # Git commit hashes
                r"branch.*[a-zA-Z-_]+",  # Git branch names
                r"tag.*[a-zA-Z-_\d\.]+",  # Git tag names
            ],
            "configuration_info": [
                r"SECRET_KEY.*=",  # Secret key configurations
                r"API_KEY.*=",  # API key configurations
                r"PASSWORD.*=",  # Password configurations
                r"TOKEN.*=",  # Token configurations
                r"DATABASE_URL.*=",  # Database URL configurations
                r"REDIS_URL.*=",  # Redis URL configurations
                r"MONGODB_URI.*=",  # MongoDB URI configurations
                r"AWS_.*=",  # AWS configurations
                r"GOOGLE_.*=",  # Google configurations
                r"AZURE_.*=",  # Azure configurations
                r"OPENAI_.*=",  # OpenAI configurations
                r"ANTHROPIC_.*=",  # Anthropic configurations
                r"config\..*=",  # Configuration object references
                r"settings\..*=",  # Settings object references
                r"env\..*=",  # Environment variable references
                r"os\.environ",  # Environment access
                r"getenv\(",  # Environment variable access
                r"environ\[",  # Environment variable access
                r"\.env",  # Environment file references
                r"config\.ini",  # Configuration file references
                r"settings\.json",  # Settings file references
                r"config\.yaml",  # YAML configuration files
                r"config\.yml",  # YAML configuration files
                r"appsettings\.json",  # .NET configuration files
                r"web\.config",  # .NET web configuration
                r"application\.properties",  # Java properties files
                r"application\.yml",  # Spring Boot configuration
            ],
            "system_info": [
                r"hostname.*:",  # System hostname
                r"server.*name.*:",  # Server name
                r"machine.*name.*:",  # Machine name
                r"computer.*name.*:",  # Computer name
                r"domain.*name.*:",  # Domain name
                r"ip.*address.*:",  # IP address
                r"mac.*address.*:",  # MAC address
                r"network.*interface.*:",  # Network interface
                r"operating.*system.*:",  # Operating system
                r"os.*version.*:",  # OS version
                r"kernel.*version.*:",  # Kernel version
                r"architecture.*:",  # System architecture
                r"cpu.*model.*:",  # CPU model
                r"memory.*size.*:",  # Memory size
                r"disk.*size.*:",  # Disk size
                r"uptime.*:",  # System uptime
                r"load.*average.*:",  # System load
                r"process.*id.*:",  # Process ID
                r"thread.*id.*:",  # Thread ID
                r"user.*id.*:",  # User ID
                r"group.*id.*:",  # Group ID
                r"session.*id.*:",  # Session ID
                r"request.*id.*:",  # Request ID (might be OK in some contexts)
                r"correlation.*id.*:",  # Correlation ID
                r"trace.*id.*:",  # Trace ID
                r"span.*id.*:",  # Span ID
            ],
        }

        # Expected generic error messages for production
        self.expected_generic_messages = {
            400: ["Bad Request", "Invalid request", "Validation error"],
            401: [
                "Unauthorized",
                "Authentication required",
                "Invalid credentials",
            ],
            403: ["Forbidden", "Access denied", "Permission denied"],
            404: ["Not Found", "Resource not found", "Endpoint not found"],
            405: ["Method Not Allowed", "Method not supported"],
            422: ["Unprocessable Entity", "Validation failed", "Invalid data"],
            429: ["Too Many Requests", "Rate limit exceeded"],
            500: [
                "Internal Server Error",
                "Server error",
                "Something went wrong",
            ],
            502: ["Bad Gateway", "Gateway error"],
            503: ["Service Unavailable", "Service temporarily unavailable"],
            504: ["Gateway Timeout", "Request timeout"],
        }

    def check_for_sensitive_information(
        self, response_text: str, response_status: int
    ) -> List[Dict[str, Any]]:
        """Check response for sensitive information disclosure."""
        findings = []

        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    findings.append(
                        {
                            "category": category,
                            "pattern": pattern,
                            "matches": matches,
                            "status_code": response_status,
                            "severity": self._get_severity(category),
                            "recommendation": self._get_recommendation(category),
                        }
                    )

        return findings

    def _get_severity(self, category: str) -> str:
        """Get severity level for information disclosure category."""
        severity_map = {
            "database_errors": "HIGH",
            "stack_traces": "HIGH",
            "debug_info": "MEDIUM",
            "internal_paths": "MEDIUM",
            "version_info": "LOW",
            "configuration_info": "CRITICAL",
            "system_info": "MEDIUM",
        }
        return severity_map.get(category, "MEDIUM")

    def _get_recommendation(self, category: str) -> str:
        """Get recommendation for information disclosure category."""
        recommendations = {
            "database_errors": "Implement generic database error handling and logging",
            "stack_traces": "Disable debug mode and implement generic error responses",
            "debug_info": "Remove debug information from production responses",
            "internal_paths": "Sanitize file paths and use relative references",
            "version_info": "Remove version information from error responses",
            "configuration_info": "Never expose configuration in error messages",
            "system_info": "Remove system information from error responses",
        }
        return recommendations.get(category, "Review and sanitize error response")

    def validate_error_message_generic(self, response_text: str, status_code: int) -> bool:
        """Validate that error message is generic and safe for production."""
        expected_messages = self.expected_generic_messages.get(status_code, [])

        # Check if response contains any expected generic message
        response_lower = response_text.lower()
        for expected in expected_messages:
            if expected.lower() in response_lower:
                return True

        # If no expected message found, check for sensitive patterns
        findings = self.check_for_sensitive_information(response_text, status_code)
        return len(findings) == 0

    def test_database_error_disclosure(self) -> List[ErrorDisclosureTestCase]:
        """Test for database error information disclosure."""
        test_cases = []

        # Test SQL injection attempts that should trigger database errors
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; SELECT * FROM information_schema.tables; --",
            "' UNION SELECT NULL, version(), NULL --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' AND (SELECT COUNT(*) FROM users) > 0 --",
            "'; DELETE FROM users WHERE id = 1; --",
            "' OR 1=1 LIMIT 1 OFFSET 1 --",
            "'; EXEC xp_cmdshell('dir'); --",
            "' OR (SELECT SUBSTRING(sys.fn_sqlvarbasetostr(HashBytes('MD5','test')),3,32)) = 'test' --",
        ]

        # Test endpoints that might interact with database
        endpoints_to_test = [
            (
                "/api/v1/agents",
                "POST",
                {"name": "TEST_PAYLOAD", "description": "test"},
            ),
            ("/api/v1/agents/search", "GET", None),
            (
                "/api/v1/auth/login",
                "POST",
                {"username": "TEST_PAYLOAD", "password": "test"},
            ),
            (
                "/api/v1/auth/register",
                "POST",
                {
                    "username": "TEST_PAYLOAD",
                    "email": "test@test.com",
                    "password": "test",
                },
            ),
            ("/api/v1/system/status", "GET", None),
            ("/api/v1/monitoring/metrics", "GET", None),
        ]

        for payload in sql_injection_payloads:
            for endpoint, method, base_data in endpoints_to_test:
                test_case = ErrorDisclosureTestCase(
                    name=f"database_error_{method}_{endpoint.replace('/', '_')}_{payload[:20]}",
                    description=f"Test database error disclosure for {method} {endpoint} with payload: {payload[:50]}",
                    category="database_errors",
                )

                try:
                    # Prepare request data
                    if base_data and isinstance(base_data, dict):
                        # Replace placeholder with payload
                        test_data = {}
                        for key, value in base_data.items():
                            if value == "TEST_PAYLOAD":
                                test_data[key] = payload
                            else:
                                test_data[key] = value
                    else:
                        test_data = None

                    # Make request
                    if method == "GET":
                        # For GET requests, add payload as query parameter
                        test_endpoint = (
                            f"{endpoint}?search={payload}" if "search" not in endpoint else endpoint
                        )
                        response = self.client.get(test_endpoint)
                    elif method == "POST":
                        response = self.client.post(endpoint, json=test_data)
                    elif method == "PUT":
                        response = self.client.put(endpoint, json=test_data)
                    elif method == "DELETE":
                        response = self.client.delete(endpoint)
                    else:
                        continue

                    # Check response for database error disclosure
                    response_text = (
                        response.text if hasattr(response, "text") else str(response.content)
                    )
                    findings = self.check_for_sensitive_information(
                        response_text, response.status_code
                    )

                    database_findings = [f for f in findings if f["category"] == "database_errors"]

                    if database_findings:
                        test_case.passed = False
                        test_case.error_details = database_findings
                        test_case.response_data = {
                            "status_code": response.status_code,
                            "response_text": response_text[:1000],  # Limit size
                            "payload": payload,
                            "endpoint": endpoint,
                            "method": method,
                        }
                    else:
                        test_case.passed = True

                except Exception as e:
                    test_case.passed = False
                    test_case.error_details = [
                        {"error": str(e), "category": "test_execution_error"}
                    ]

                test_cases.append(test_case)

        return test_cases

    def test_exception_handling_disclosure(
        self,
    ) -> List[ErrorDisclosureTestCase]:
        """Test for exception and stack trace disclosure."""
        test_cases = []

        # Test various exception-triggering scenarios
        exception_payloads = [
            # Invalid JSON
            ("invalid_json", '{"invalid": json}', "application/json"),
            # Large payload
            ("large_payload", "A" * 100000, "text/plain"),
            # Invalid content type
            ("invalid_content_type", "test", "application/xml"),
            # Malformed data
            ("malformed_data", {"key": object()}, "application/json"),
            # Unicode issues
            ("unicode_issue", "test\x00\x01\x02", "text/plain"),
            # Null bytes
            ("null_bytes", "test\x00null", "text/plain"),
        ]

        endpoints_to_test = [
            "/api/v1/agents",
            "/api/v1/auth/login",
            "/api/v1/system/status",
            "/api/v1/monitoring/metrics",
        ]

        for payload_name, payload_data, content_type in exception_payloads:
            for endpoint in endpoints_to_test:
                test_case = ErrorDisclosureTestCase(
                    name=f"exception_handling_{endpoint.replace('/', '_')}_{payload_name}",
                    description=f"Test exception handling for {endpoint} with {payload_name}",
                    category="exception_handling",
                )

                try:
                    # Make request with malformed data
                    if isinstance(payload_data, str):
                        response = self.client.post(
                            endpoint,
                            data=payload_data,
                            headers={"Content-Type": content_type},
                        )
                    else:
                        response = self.client.post(endpoint, json=payload_data)

                    # Check response for stack trace disclosure
                    response_text = (
                        response.text if hasattr(response, "text") else str(response.content)
                    )
                    findings = self.check_for_sensitive_information(
                        response_text, response.status_code
                    )

                    stack_trace_findings = [f for f in findings if f["category"] == "stack_traces"]
                    debug_findings = [f for f in findings if f["category"] == "debug_info"]

                    if stack_trace_findings or debug_findings:
                        test_case.passed = False
                        test_case.error_details = stack_trace_findings + debug_findings
                        test_case.response_data = {
                            "status_code": response.status_code,
                            "response_text": response_text[:1000],
                            "payload_name": payload_name,
                            "endpoint": endpoint,
                        }
                    else:
                        test_case.passed = True

                except Exception as e:
                    test_case.passed = False
                    test_case.error_details = [
                        {"error": str(e), "category": "test_execution_error"}
                    ]

                test_cases.append(test_case)

        return test_cases

    def test_path_disclosure(self) -> List[ErrorDisclosureTestCase]:
        """Test for internal path disclosure."""
        test_cases = []

        # Test file path traversal attempts
        path_traversal_payloads = [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "/home/user/.ssh/id_rsa",
            "C:\\Program Files\\",
            "/var/log/",
            "/tmp/",
            "~/.bashrc",
            "/proc/version",
            "/sys/kernel/",
        ]

        # Test endpoints that might handle file operations
        file_endpoints = [
            ("/api/v1/agents", "POST"),
            ("/api/v1/system/logs", "GET"),
            ("/api/v1/monitoring/export", "GET"),
        ]

        for payload in path_traversal_payloads:
            for endpoint, method in file_endpoints:
                test_case = ErrorDisclosureTestCase(
                    name=f"path_disclosure_{method}_{endpoint.replace('/', '_')}_{payload[:20]}",
                    description=f"Test path disclosure for {method} {endpoint} with payload: {payload}",
                    category="path_disclosure",
                )

                try:
                    if method == "GET":
                        response = self.client.get(f"{endpoint}?file={payload}")
                    else:
                        response = self.client.post(endpoint, json={"file": payload})

                    response_text = (
                        response.text if hasattr(response, "text") else str(response.content)
                    )
                    findings = self.check_for_sensitive_information(
                        response_text, response.status_code
                    )

                    path_findings = [f for f in findings if f["category"] == "internal_paths"]

                    if path_findings:
                        test_case.passed = False
                        test_case.error_details = path_findings
                        test_case.response_data = {
                            "status_code": response.status_code,
                            "response_text": response_text[:1000],
                            "payload": payload,
                            "endpoint": endpoint,
                            "method": method,
                        }
                    else:
                        test_case.passed = True

                except Exception as e:
                    test_case.passed = False
                    test_case.error_details = [
                        {"error": str(e), "category": "test_execution_error"}
                    ]

                test_cases.append(test_case)

        return test_cases

    def test_version_information_disclosure(
        self,
    ) -> List[ErrorDisclosureTestCase]:
        """Test for version information disclosure."""
        test_cases = []

        # Test common endpoints that might expose version info
        version_test_endpoints = [
            "/",
            "/health",
            "/api/v1",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/system/status",
            "/api/v1/system/info",
            "/api/v1/monitoring/metrics",
        ]

        for endpoint in version_test_endpoints:
            test_case = ErrorDisclosureTestCase(
                name=f"version_disclosure_{endpoint.replace('/', '_')}",
                description=f"Test version information disclosure for {endpoint}",
                category="version_disclosure",
            )

            try:
                response = self.client.get(endpoint)
                response_text = (
                    response.text if hasattr(response, "text") else str(response.content)
                )
                findings = self.check_for_sensitive_information(response_text, response.status_code)

                version_findings = [f for f in findings if f["category"] == "version_info"]

                # Version info in some endpoints might be acceptable (like /health)
                # But we'll flag it for review
                if version_findings and endpoint not in ["/", "/health"]:
                    test_case.passed = False
                    test_case.error_details = version_findings
                    test_case.response_data = {
                        "status_code": response.status_code,
                        "response_text": response_text[:1000],
                        "endpoint": endpoint,
                    }
                else:
                    test_case.passed = True

            except Exception as e:
                test_case.passed = False
                test_case.error_details = [{"error": str(e), "category": "test_execution_error"}]

            test_cases.append(test_case)

        return test_cases

    def test_configuration_disclosure(self) -> List[ErrorDisclosureTestCase]:
        """Test for configuration information disclosure."""
        test_cases = []

        # Test for configuration disclosure in error responses
        config_test_scenarios = [
            ("invalid_database_connection", {"database_url": "invalid://url"}),
            ("invalid_redis_connection", {"redis_url": "invalid://redis"}),
            ("invalid_api_key", {"api_key": "invalid_key"}),
            ("missing_environment_variable", {"env_var": "MISSING_ENV_VAR"}),
        ]

        config_endpoints = [
            "/api/v1/system/config",
            "/api/v1/system/health",
            "/api/v1/monitoring/config",
        ]

        for scenario_name, test_data in config_test_scenarios:
            for endpoint in config_endpoints:
                test_case = ErrorDisclosureTestCase(
                    name=f"config_disclosure_{endpoint.replace('/', '_')}_{scenario_name}",
                    description=f"Test configuration disclosure for {endpoint} with {scenario_name}",
                    category="configuration_disclosure",
                )

                try:
                    response = self.client.post(endpoint, json=test_data)
                    response_text = (
                        response.text if hasattr(response, "text") else str(response.content)
                    )
                    findings = self.check_for_sensitive_information(
                        response_text, response.status_code
                    )

                    config_findings = [f for f in findings if f["category"] == "configuration_info"]

                    if config_findings:
                        test_case.passed = False
                        test_case.error_details = config_findings
                        test_case.response_data = {
                            "status_code": response.status_code,
                            "response_text": response_text[:1000],
                            "scenario": scenario_name,
                            "endpoint": endpoint,
                        }
                    else:
                        test_case.passed = True

                except Exception as e:
                    test_case.passed = False
                    test_case.error_details = [
                        {"error": str(e), "category": "test_execution_error"}
                    ]

                test_cases.append(test_case)

        return test_cases

    def test_http_status_consistency(self) -> List[ErrorDisclosureTestCase]:
        """Test HTTP status code consistency and proper error handling."""
        test_cases = []

        # Test status code consistency scenarios
        status_test_scenarios = [
            ("non_existent_endpoint", "GET", "/api/v1/nonexistent", None, 404),
            ("invalid_method", "PATCH", "/api/v1/agents", None, 405),
            (
                "malformed_json",
                "POST",
                "/api/v1/agents",
                '{"invalid": json}',
                400,
            ),
            ("missing_auth", "GET", "/api/v1/agents/protected", None, 401),
            (
                "insufficient_permissions",
                "DELETE",
                "/api/v1/system/shutdown",
                None,
                403,
            ),
            (
                "too_large_payload",
                "POST",
                "/api/v1/agents",
                "A" * 1000000,
                413,
            ),
        ]

        for (
            scenario_name,
            method,
            endpoint,
            data,
            expected_status,
        ) in status_test_scenarios:
            test_case = ErrorDisclosureTestCase(
                name=f"status_consistency_{scenario_name}",
                description=f"Test status code consistency for {scenario_name}",
                category="status_consistency",
            )

            try:
                if method == "GET":
                    response = self.client.get(endpoint)
                elif method == "POST":
                    if data and data.startswith("{"):
                        response = self.client.post(
                            endpoint,
                            data=data,
                            headers={"Content-Type": "application/json"},
                        )
                    else:
                        response = self.client.post(endpoint, json={"data": data} if data else {})
                elif method == "DELETE":
                    response = self.client.delete(endpoint)
                elif method == "PATCH":
                    response = self.client.patch(endpoint, json={})
                else:
                    response = self.client.request(method, endpoint)

                # Check if status code matches expected
                if response.status_code == expected_status:
                    test_case.passed = True
                else:
                    test_case.passed = False
                    test_case.error_details = [
                        {
                            "expected_status": expected_status,
                            "actual_status": response.status_code,
                            "category": "status_mismatch",
                        }
                    ]

                # Also check for information disclosure in error response
                response_text = (
                    response.text if hasattr(response, "text") else str(response.content)
                )
                findings = self.check_for_sensitive_information(response_text, response.status_code)

                if findings:
                    test_case.passed = False
                    if not test_case.error_details:
                        test_case.error_details = []
                    test_case.error_details.extend(findings)

                test_case.response_data = {
                    "status_code": response.status_code,
                    "response_text": response_text[:500],
                    "scenario": scenario_name,
                    "expected_status": expected_status,
                }

            except Exception as e:
                test_case.passed = False
                test_case.error_details = [{"error": str(e), "category": "test_execution_error"}]

            test_cases.append(test_case)

        return test_cases

    def test_error_message_sanitization(self) -> List[ErrorDisclosureTestCase]:
        """Test that error messages are properly sanitized."""
        test_cases = []

        # Test various input sanitization scenarios
        sanitization_scenarios = [
            ("xss_attempt", "<script>alert('xss')</script>"),
            ("sql_injection", "' OR '1'='1' --"),
            ("path_traversal", "../../../../etc/passwd"),
            ("command_injection", "; ls -la"),
            ("null_bytes", "test\x00null"),
            ("unicode_attack", "test\u202e\u202d"),
            ("format_string", "%s%s%s%s"),
            ("buffer_overflow", "A" * 10000),
        ]

        sanitization_endpoints = [
            "/api/v1/agents",
            "/api/v1/auth/login",
            "/api/v1/system/search",
        ]

        for scenario_name, payload in sanitization_scenarios:
            for endpoint in sanitization_endpoints:
                test_case = ErrorDisclosureTestCase(
                    name=f"sanitization_{endpoint.replace('/', '_')}_{scenario_name}",
                    description=f"Test error message sanitization for {endpoint} with {scenario_name}",
                    category="message_sanitization",
                )

                try:
                    # Test in different request fields
                    test_data = {
                        "name": payload,
                        "description": payload,
                        "search": payload,
                        "username": payload,
                        "password": payload,
                    }

                    response = self.client.post(endpoint, json=test_data)
                    response_text = (
                        response.text if hasattr(response, "text") else str(response.content)
                    )

                    # Check if payload appears unsanitized in response
                    if payload in response_text:
                        test_case.passed = False
                        test_case.error_details = [
                            {
                                "category": "unsanitized_input",
                                "payload": payload,
                                "found_in_response": True,
                            }
                        ]
                    else:
                        test_case.passed = True

                    # Also check for other information disclosure
                    findings = self.check_for_sensitive_information(
                        response_text, response.status_code
                    )
                    if findings:
                        test_case.passed = False
                        if not test_case.error_details:
                            test_case.error_details = []
                        test_case.error_details.extend(findings)

                    test_case.response_data = {
                        "status_code": response.status_code,
                        "response_text": response_text[:500],
                        "payload": payload,
                        "scenario": scenario_name,
                        "endpoint": endpoint,
                    }

                except Exception as e:
                    test_case.passed = False
                    test_case.error_details = [
                        {"error": str(e), "category": "test_execution_error"}
                    ]

                test_cases.append(test_case)

        return test_cases

    def test_production_hardening(self) -> List[ErrorDisclosureTestCase]:
        """Test production hardening and security configuration."""
        test_cases = []

        # Test debug mode detection
        debug_test = ErrorDisclosureTestCase(
            name="debug_mode_disabled",
            description="Verify debug mode is disabled in production",
            category="production_hardening",
        )

        try:
            # Check various endpoints for debug indicators
            debug_endpoints = ["/", "/docs", "/api/v1/system/status"]
            debug_found = False

            for endpoint in debug_endpoints:
                response = self.client.get(endpoint)
                response_text = (
                    response.text if hasattr(response, "text") else str(response.content)
                )

                debug_indicators = [
                    "debug.*true",
                    "debug.*mode",
                    "development.*mode",
                    "test.*mode",
                    "verbose.*true",
                    "traceback",
                    "__debug__",
                ]

                for indicator in debug_indicators:
                    if re.search(indicator, response_text, re.IGNORECASE):
                        debug_found = True
                        debug_test.error_details = [
                            {
                                "category": "debug_mode_enabled",
                                "indicator": indicator,
                                "endpoint": endpoint,
                            }
                        ]
                        break

                if debug_found:
                    break

            debug_test.passed = not debug_found

        except Exception as e:
            debug_test.passed = False
            debug_test.error_details = [{"error": str(e), "category": "test_execution_error"}]

        test_cases.append(debug_test)

        # Test error logging vs display separation
        logging_test = ErrorDisclosureTestCase(
            name="error_logging_separation",
            description="Verify errors are logged but not displayed in detail",
            category="production_hardening",
        )

        try:
            # Trigger an error and check if detailed logging occurs without disclosure
            response = self.client.post("/api/v1/agents", json={"invalid": "data"})
            response_text = response.text if hasattr(response, "text") else str(response.content)

            # Should have generic error message
            generic_message = self.validate_error_message_generic(
                response_text, response.status_code
            )

            if generic_message:
                logging_test.passed = True
            else:
                logging_test.passed = False
                logging_test.error_details = [{"category": "detailed_error_exposed"}]
                logging_test.response_data = {
                    "status_code": response.status_code,
                    "response_text": response_text[:500],
                }

        except Exception as e:
            logging_test.passed = False
            logging_test.error_details = [{"error": str(e), "category": "test_execution_error"}]

        test_cases.append(logging_test)

        # Test security headers presence
        headers_test = ErrorDisclosureTestCase(
            name="security_headers_present",
            description="Verify security headers are present in error responses",
            category="production_hardening",
        )

        try:
            response = self.client.get("/api/v1/nonexistent")

            required_security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Referrer-Policy",
                "Content-Security-Policy",
            ]

            missing_headers = []
            for header in required_security_headers:
                if header not in response.headers:
                    missing_headers.append(header)

            if missing_headers:
                headers_test.passed = False
                headers_test.error_details = [
                    {
                        "category": "missing_security_headers",
                        "missing_headers": missing_headers,
                    }
                ]
            else:
                headers_test.passed = True

        except Exception as e:
            headers_test.passed = False
            headers_test.error_details = [{"error": str(e), "category": "test_execution_error"}]

        test_cases.append(headers_test)

        return test_cases

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all error handling disclosure tests."""
        self.logger.info("Starting comprehensive error handling information disclosure tests")

        # Run all test categories
        all_test_results = []

        # Database error disclosure tests
        self.logger.info("Running database error disclosure tests...")
        db_tests = self.test_database_error_disclosure()
        all_test_results.extend(db_tests)

        # Exception handling tests
        self.logger.info("Running exception handling disclosure tests...")
        exception_tests = self.test_exception_handling_disclosure()
        all_test_results.extend(exception_tests)

        # Path disclosure tests
        self.logger.info("Running path disclosure tests...")
        path_tests = self.test_path_disclosure()
        all_test_results.extend(path_tests)

        # Version information tests
        self.logger.info("Running version information disclosure tests...")
        version_tests = self.test_version_information_disclosure()
        all_test_results.extend(version_tests)

        # Configuration disclosure tests
        self.logger.info("Running configuration disclosure tests...")
        config_tests = self.test_configuration_disclosure()
        all_test_results.extend(config_tests)

        # HTTP status consistency tests
        self.logger.info("Running HTTP status consistency tests...")
        status_tests = self.test_http_status_consistency()
        all_test_results.extend(status_tests)

        # Error message sanitization tests
        self.logger.info("Running error message sanitization tests...")
        sanitization_tests = self.test_error_message_sanitization()
        all_test_results.extend(sanitization_tests)

        # Production hardening tests
        self.logger.info("Running production hardening tests...")
        hardening_tests = self.test_production_hardening()
        all_test_results.extend(hardening_tests)

        # Compile results
        total_tests = len(all_test_results)
        passed_tests = len([t for t in all_test_results if t.passed])
        failed_tests = total_tests - passed_tests

        # Categorize failures by severity
        critical_failures = []
        high_failures = []
        medium_failures = []
        low_failures = []

        for test in all_test_results:
            if not test.passed and test.error_details:
                for error in test.error_details:
                    severity = error.get("severity", "MEDIUM")
                    if severity == "CRITICAL":
                        critical_failures.append(test)
                    elif severity == "HIGH":
                        high_failures.append(test)
                    elif severity == "MEDIUM":
                        medium_failures.append(test)
                    else:
                        low_failures.append(test)

        # Generate comprehensive report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "critical_failures": len(critical_failures),
                "high_failures": len(high_failures),
                "medium_failures": len(medium_failures),
                "low_failures": len(low_failures),
            },
            "test_categories": {
                "database_errors": len(
                    [t for t in all_test_results if t.category == "database_errors"]
                ),
                "exception_handling": len(
                    [t for t in all_test_results if t.category == "exception_handling"]
                ),
                "path_disclosure": len(
                    [t for t in all_test_results if t.category == "path_disclosure"]
                ),
                "version_disclosure": len(
                    [t for t in all_test_results if t.category == "version_disclosure"]
                ),
                "configuration_disclosure": len(
                    [t for t in all_test_results if t.category == "configuration_disclosure"]
                ),
                "status_consistency": len(
                    [t for t in all_test_results if t.category == "status_consistency"]
                ),
                "message_sanitization": len(
                    [t for t in all_test_results if t.category == "message_sanitization"]
                ),
                "production_hardening": len(
                    [t for t in all_test_results if t.category == "production_hardening"]
                ),
            },
            "failures": {
                "critical": [self._test_to_dict(t) for t in critical_failures],
                "high": [self._test_to_dict(t) for t in high_failures],
                "medium": [self._test_to_dict(t) for t in medium_failures],
                "low": [self._test_to_dict(t) for t in low_failures],
            },
            "recommendations": self._generate_recommendations(all_test_results),
            "detailed_results": [self._test_to_dict(t) for t in all_test_results],
        }

        self.logger.info(
            f"Error handling disclosure tests completed: {passed_tests}/{total_tests} passed"
        )

        return report

    def _test_to_dict(self, test_case: ErrorDisclosureTestCase) -> Dict[str, Any]:
        """Convert test case to dictionary for JSON serialization."""
        return {
            "name": test_case.name,
            "description": test_case.description,
            "category": test_case.category,
            "passed": test_case.passed,
            "error_details": test_case.error_details,
            "response_data": test_case.response_data,
        }

    def _generate_recommendations(self, test_results: List[ErrorDisclosureTestCase]) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []

        failed_tests = [t for t in test_results if not t.passed]

        if failed_tests:
            recommendations.append(
                "CRITICAL: Information disclosure vulnerabilities detected in error handling"
            )

            # Check for specific categories
            categories_with_failures = set()
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        categories_with_failures.add(error.get("category", test.category))

            if "database_errors" in categories_with_failures:
                recommendations.append(
                    "- Implement generic database error handling to prevent SQL error disclosure"
                )
                recommendations.append("- Use parameterized queries and input validation")
                recommendations.append(
                    "- Log detailed errors server-side but return generic messages to clients"
                )

            if "stack_traces" in categories_with_failures:
                recommendations.append("- Disable debug mode in production environment")
                recommendations.append(
                    "- Implement custom exception handlers that return generic error messages"
                )
                recommendations.append("- Use structured logging for internal error tracking")

            if "internal_paths" in categories_with_failures:
                recommendations.append("- Sanitize file paths in error messages")
                recommendations.append(
                    "- Use relative paths or abstract identifiers in client responses"
                )
                recommendations.append("- Implement path traversal protection")

            if "version_info" in categories_with_failures:
                recommendations.append("- Remove version information from error responses")
                recommendations.append("- Implement custom error pages without server signatures")

            if "configuration_info" in categories_with_failures:
                recommendations.append(
                    "- URGENT: Never expose configuration details in error messages"
                )
                recommendations.append("- Review and sanitize all error response content")
                recommendations.append("- Implement configuration validation without disclosure")

            recommendations.append("- Implement comprehensive error handling middleware")
            recommendations.append("- Use centralized logging with correlation IDs")
            recommendations.append("- Regular security testing of error handling paths")
            recommendations.append("- Implement proper security headers on all responses")

        else:
            recommendations.append("No critical information disclosure vulnerabilities detected")
            recommendations.append("Continue regular security testing and monitoring")

        return recommendations


class TestErrorHandlingInformationDisclosure:
    """pytest test class for error handling information disclosure tests."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def error_tester(self, client):
        """Create error handling tester."""
        return ErrorHandlingTester(client)

    def test_database_error_disclosure(self, error_tester):
        """Test database error information disclosure."""
        results = error_tester.test_database_error_disclosure()

        # Check if any tests failed
        failed_tests = [test for test in results if not test.passed]

        if failed_tests:
            failure_details = []
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        if error.get("category") == "database_errors":
                            failure_details.append(
                                f"Database error disclosed in {test.name}: {error}"
                            )

            if failure_details:
                pytest.fail(f"Database error disclosure detected: {failure_details}")

    def test_exception_handling_disclosure(self, error_tester):
        """Test exception handling disclosure."""
        results = error_tester.test_exception_handling_disclosure()

        failed_tests = [test for test in results if not test.passed]

        if failed_tests:
            failure_details = []
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        if error.get("category") in [
                            "stack_traces",
                            "debug_info",
                        ]:
                            failure_details.append(
                                f"Exception details disclosed in {test.name}: {error}"
                            )

            if failure_details:
                pytest.fail(f"Exception handling disclosure detected: {failure_details}")

    def test_path_disclosure(self, error_tester):
        """Test internal path disclosure."""
        results = error_tester.test_path_disclosure()

        failed_tests = [test for test in results if not test.passed]

        if failed_tests:
            failure_details = []
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        if error.get("category") == "internal_paths":
                            failure_details.append(
                                f"Internal path disclosed in {test.name}: {error}"
                            )

            if failure_details:
                pytest.fail(f"Internal path disclosure detected: {failure_details}")

    def test_configuration_disclosure(self, error_tester):
        """Test configuration information disclosure."""
        results = error_tester.test_configuration_disclosure()

        failed_tests = [test for test in results if not test.passed]

        if failed_tests:
            failure_details = []
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        if error.get("category") == "configuration_info":
                            failure_details.append(
                                f"Configuration info disclosed in {test.name}: {error}"
                            )

            if failure_details:
                pytest.fail(f"CRITICAL: Configuration disclosure detected: {failure_details}")

    def test_http_status_consistency(self, error_tester):
        """Test HTTP status code consistency."""
        results = error_tester.test_http_status_consistency()

        failed_tests = [test for test in results if not test.passed]

        if failed_tests:
            failure_details = []
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        if error.get("category") == "status_mismatch":
                            failure_details.append(
                                f"Status code mismatch in {test.name}: expected {error.get('expected_status')}, got {error.get('actual_status')}"
                            )

            if failure_details:
                pytest.fail(f"HTTP status consistency issues: {failure_details}")

    def test_error_message_sanitization(self, error_tester):
        """Test error message sanitization."""
        results = error_tester.test_error_message_sanitization()

        failed_tests = [test for test in results if not test.passed]

        if failed_tests:
            failure_details = []
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        if error.get("category") == "unsanitized_input":
                            failure_details.append(
                                f"Unsanitized input in response for {test.name}: {error}"
                            )

            if failure_details:
                pytest.fail(f"Error message sanitization failures: {failure_details}")

    def test_production_hardening(self, error_tester):
        """Test production hardening configuration."""
        results = error_tester.test_production_hardening()

        failed_tests = [test for test in results if not test.passed]

        if failed_tests:
            failure_details = []
            for test in failed_tests:
                if test.error_details:
                    for error in test.error_details:
                        category = error.get("category")
                        if category in [
                            "debug_mode_enabled",
                            "detailed_error_exposed",
                            "missing_security_headers",
                        ]:
                            failure_details.append(
                                f"Production hardening issue in {test.name}: {error}"
                            )

            if failure_details:
                pytest.fail(f"Production hardening failures: {failure_details}")

    def test_comprehensive_error_disclosure_scan(self, error_tester):
        """Run comprehensive error disclosure scan."""
        report = error_tester.run_all_tests()

        # Check overall results
        summary = report["summary"]

        # Fail if critical or high severity issues found
        if summary["critical_failures"] > 0:
            pytest.fail(
                f"CRITICAL: {summary['critical_failures']} critical information disclosure vulnerabilities detected"
            )

        if summary["high_failures"] > 0:
            pytest.fail(
                f"HIGH: {summary['high_failures']} high-severity information disclosure vulnerabilities detected"
            )

        # Warn about medium severity issues
        if summary["medium_failures"] > 0:
            print(
                f"WARNING: {summary['medium_failures']} medium-severity information disclosure issues detected"
            )

        # Log recommendations
        if report["recommendations"]:
            print("Security Recommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

        # Ensure minimum pass rate
        if summary["pass_rate"] < 95.0:
            pytest.fail(
                f"Error handling security test pass rate too low: {summary['pass_rate']:.1f}% (minimum 95%)"
            )


if __name__ == "__main__":
    """Direct execution for standalone testing."""
    client = TestClient(app)
    tester = ErrorHandlingTester(client)

    print("Running comprehensive error handling information disclosure tests...")
    report = tester.run_all_tests()

    # Print summary
    print(f"\n{'='*80}")
    print("ERROR HANDLING INFORMATION DISCLOSURE TEST REPORT")
    print(f"{'='*80}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Pass Rate: {report['summary']['pass_rate']:.1f}%")
    print("\nSeverity Breakdown:")
    print(f"  Critical: {report['summary']['critical_failures']}")
    print(f"  High: {report['summary']['high_failures']}")
    print(f"  Medium: {report['summary']['medium_failures']}")
    print(f"  Low: {report['summary']['low_failures']}")

    # Print recommendations
    if report["recommendations"]:
        print(f"\n{'='*50}")
        print("SECURITY RECOMMENDATIONS")
        print(f"{'='*50}")
        for rec in report["recommendations"]:
            print(f"• {rec}")

    # Save detailed report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = (
        f"/home/green/FreeAgentics/tests/security/error_disclosure_report_{timestamp}.json"
    )

    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Exit with error code if critical issues found
    if report["summary"]["critical_failures"] > 0:
        exit(1)
    elif report["summary"]["high_failures"] > 0:
        exit(2)
    else:
        exit(0)
