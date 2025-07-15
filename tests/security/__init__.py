"""
FreeAgentics Security Testing Framework

This package provides a comprehensive penetration testing framework for the FreeAgentics platform,
including authentication bypass testing, session management vulnerabilities, authorization flaws,
API security issues, and business logic bypasses.

The framework is designed to:
- Validate security against real-world attack scenarios
- Document all findings with proof-of-concept and remediation steps
- Generate comprehensive reports in multiple formats
- Integrate with existing security infrastructure
- Provide safe execution with rollback capabilities

Example usage:
    from tests.security import PenetrationTestRunner

    runner = PenetrationTestRunner()
    results = await runner.run_all_tests()
    print(f"Found {len(results['detailed_findings'])} vulnerabilities")

For CLI usage:
    python -m tests.security.penetration_test_runner --module all --output html json
"""

from .api_security_tests import APISecurityTests
from .authentication_bypass_tests import AuthenticationBypassTests
from .authorization_tests import AuthorizationTests
from .business_logic_tests import BusinessLogicTests
from .penetration_test_runner import PenetrationTestRunner
from .penetration_testing_framework import (
    BasePenetrationTest,
    PenetrationTestingFramework,
    SeverityLevel,
    TestResult,
    VulnerabilityFinding,
    VulnerabilityType,
)
from .session_management_tests import SessionManagementTests

__version__ = "1.0.0"
__author__ = "FreeAgentics Security Team"

__all__ = [
    "PenetrationTestingFramework",
    "PenetrationTestRunner",
    "BasePenetrationTest",
    "VulnerabilityFinding",
    "VulnerabilityType",
    "SeverityLevel",
    "TestResult",
    "AuthenticationBypassTests",
    "SessionManagementTests",
    "AuthorizationTests",
    "APISecurityTests",
    "BusinessLogicTests",
]
