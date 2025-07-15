"""
Comprehensive Penetration Testing Framework for FreeAgentics Platform

This framework provides production-ready penetration testing capabilities that validate
security against real-world attack scenarios while documenting all findings with
proof-of-concept and remediation steps.

Architecture:
- Modular testing components for different attack vectors
- Automated vulnerability discovery and exploitation
- Comprehensive reporting with remediation guidance
- Integration with existing security infrastructure
- Safe execution with rollback capabilities
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi.testclient import TestClient

from api.main import app
from auth.security_implementation import AuthenticationManager, UserRole

# Configure logging for penetration testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VulnerabilityType(str, Enum):
    """Types of vulnerabilities that can be discovered."""

    AUTHENTICATION_BYPASS = "authentication_bypass"
    SQL_INJECTION = "sql_injection"
    NOSQL_INJECTION = "nosql_injection"
    LDAP_INJECTION = "ldap_injection"
    JWT_MANIPULATION = "jwt_manipulation"
    SESSION_FIXATION = "session_fixation"
    SESSION_HIJACKING = "session_hijacking"
    CSRF = "csrf"
    PRIVILEGE_ESCALATION_HORIZONTAL = "privilege_escalation_horizontal"
    PRIVILEGE_ESCALATION_VERTICAL = "privilege_escalation_vertical"
    IDOR = "idor"
    PARAMETER_POLLUTION = "parameter_pollution"
    HTTP_METHOD_TAMPERING = "http_method_tampering"
    RATE_LIMITING_BYPASS = "rate_limiting_bypass"
    BUSINESS_LOGIC_BYPASS = "business_logic_bypass"
    RACE_CONDITION = "race_condition"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"


class SeverityLevel(str, Enum):
    """Severity levels for vulnerabilities."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class VulnerabilityFinding:
    """Represents a vulnerability finding from penetration testing."""

    vulnerability_type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    affected_endpoint: str
    proof_of_concept: str
    exploitation_steps: List[str]
    remediation_steps: List[str]
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    references: List[str] = None
    discovered_at: datetime = None
    test_method: str = ""

    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.now(timezone.utc)
        if self.references is None:
            self.references = []


@dataclass
class TestResult:
    """Result of a penetration test."""

    test_name: str
    success: bool
    vulnerabilities: List[VulnerabilityFinding]
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePenetrationTest(ABC):
    """Base class for all penetration tests."""

    def __init__(self, client: TestClient, auth_manager: AuthenticationManager):
        self.client = client
        self.auth_manager = auth_manager
        self.vulnerabilities: List[VulnerabilityFinding] = []

    @abstractmethod
    async def execute(self) -> TestResult:
        """Execute the penetration test."""

    def add_vulnerability(self, vulnerability: VulnerabilityFinding):
        """Add a discovered vulnerability."""
        self.vulnerabilities.append(vulnerability)
        logger.warning(f"Vulnerability discovered: {vulnerability.title}")

    def create_test_user(self, role: UserRole = UserRole.OBSERVER) -> Tuple[str, str, str]:
        """Create a test user for testing purposes."""
        username = f"test_user_{int(time.time())}"
        password = "test_password_123"
        email = f"{username}@test.com"

        user = self.auth_manager.register_user(username, email, password, role)
        return username, password, user.user_id

    def get_auth_token(self, username: str, password: str) -> str:
        """Get authentication token for a user."""
        user = self.auth_manager.authenticate_user(username, password)
        if not user:
            raise ValueError("Authentication failed")
        return self.auth_manager.create_access_token(user)

    def get_auth_headers(self, token: str) -> Dict[str, str]:
        """Get authentication headers."""
        return {"Authorization": f"Bearer {token}"}


class PenetrationTestingFramework:
    """Main framework for orchestrating penetration tests."""

    def __init__(self):
        self.client = TestClient(app)
        self.auth_manager = AuthenticationManager()
        self.test_modules: List[BasePenetrationTest] = []
        self.results: List[TestResult] = []

    def register_test_module(self, test_module: BasePenetrationTest):
        """Register a penetration test module."""
        self.test_modules.append(test_module)

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered penetration tests."""
        logger.info("Starting comprehensive penetration testing suite")
        start_time = time.time()

        all_vulnerabilities = []
        test_results = []

        for test_module in self.test_modules:
            try:
                logger.info(f"Executing test: {test_module.__class__.__name__}")
                result = await test_module.execute()
                test_results.append(result)
                all_vulnerabilities.extend(result.vulnerabilities)

            except Exception as e:
                logger.error(f"Test {test_module.__class__.__name__} failed: {e}")
                test_results.append(
                    TestResult(
                        test_name=test_module.__class__.__name__,
                        success=False,
                        vulnerabilities=[],
                        execution_time=0,
                        error_message=str(e),
                    )
                )

        total_time = time.time() - start_time

        # Generate comprehensive report
        report = self._generate_report(test_results, all_vulnerabilities, total_time)

        logger.info(f"Penetration testing completed in {total_time:.2f} seconds")
        logger.info(f"Total vulnerabilities found: {len(all_vulnerabilities)}")

        return report

    def _generate_report(
        self,
        test_results: List[TestResult],
        vulnerabilities: List[VulnerabilityFinding],
        total_time: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive penetration testing report."""

        # Categorize vulnerabilities by severity
        severity_counts = {severity.value: 0 for severity in SeverityLevel}
        for vuln in vulnerabilities:
            severity_counts[vuln.severity.value] += 1

        # Categorize by vulnerability type
        type_counts = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.vulnerability_type.value
            type_counts[vuln_type] = type_counts.get(vuln_type, 0) + 1

        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities)

        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "framework_version": "1.0.0",
                "target": "FreeAgentics Platform",
                "total_execution_time": total_time,
                "tests_executed": len(test_results),
                "tests_successful": sum(1 for r in test_results if r.success),
            },
            "executive_summary": {
                "total_vulnerabilities": len(vulnerabilities),
                "severity_distribution": severity_counts,
                "vulnerability_types": type_counts,
                "risk_score": risk_score,
                "recommendations": self._generate_recommendations(vulnerabilities),
            },
            "detailed_findings": [asdict(vuln) for vuln in vulnerabilities],
            "test_results": [asdict(result) for result in test_results],
            "remediation_plan": self._generate_remediation_plan(vulnerabilities),
        }

        return report

    def _calculate_risk_score(self, vulnerabilities: List[VulnerabilityFinding]) -> float:
        """Calculate overall risk score (0-100)."""
        if not vulnerabilities:
            return 0.0

        severity_weights = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 7,
            SeverityLevel.MEDIUM: 4,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1,
        }

        total_score = sum(severity_weights.get(vuln.severity, 1) for vuln in vulnerabilities)
        max_possible_score = len(vulnerabilities) * severity_weights[SeverityLevel.CRITICAL]

        return (
            min(100.0, (total_score / max_possible_score) * 100) if max_possible_score > 0 else 0.0
        )

    def _generate_recommendations(self, vulnerabilities: List[VulnerabilityFinding]) -> List[str]:
        """Generate high-level security recommendations."""
        recommendations = []

        vuln_types = {vuln.vulnerability_type for vuln in vulnerabilities}

        if VulnerabilityType.AUTHENTICATION_BYPASS in vuln_types:
            recommendations.append("Implement multi-factor authentication for all user accounts")

        if VulnerabilityType.SQL_INJECTION in vuln_types:
            recommendations.append(
                "Use parameterized queries and input validation for all database operations"
            )

        if VulnerabilityType.JWT_MANIPULATION in vuln_types:
            recommendations.append(
                "Strengthen JWT implementation with proper validation and key management"
            )

        if VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL in vuln_types:
            recommendations.append("Review and strengthen role-based access control implementation")

        if VulnerabilityType.IDOR in vuln_types:
            recommendations.append("Implement proper authorization checks for all resource access")

        critical_count = sum(
            1 for vuln in vulnerabilities if vuln.severity == SeverityLevel.CRITICAL
        )
        if critical_count > 0:
            recommendations.append(f"Immediately address {critical_count} critical vulnerabilities")

        return recommendations

    def _generate_remediation_plan(
        self, vulnerabilities: List[VulnerabilityFinding]
    ) -> Dict[str, Any]:
        """Generate detailed remediation plan."""

        # Group by severity for prioritization
        by_severity = {}
        for vuln in vulnerabilities:
            severity = vuln.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(vuln)

        # Generate timeline based on severity
        timeline = {
            "immediate": by_severity.get(SeverityLevel.CRITICAL.value, []),
            "within_week": by_severity.get(SeverityLevel.HIGH.value, []),
            "within_month": by_severity.get(SeverityLevel.MEDIUM.value, []),
            "within_quarter": by_severity.get(SeverityLevel.LOW.value, [])
            + by_severity.get(SeverityLevel.INFO.value, []),
        }

        return {
            "prioritization": timeline,
            "estimated_effort": self._estimate_remediation_effort(vulnerabilities),
            "resource_requirements": self._estimate_resource_requirements(vulnerabilities),
        }

    def _estimate_remediation_effort(
        self, vulnerabilities: List[VulnerabilityFinding]
    ) -> Dict[str, str]:
        """Estimate effort required for remediation."""
        effort_mapping = {
            VulnerabilityType.SQL_INJECTION: "Medium - Requires code review and parameterized queries",
            VulnerabilityType.AUTHENTICATION_BYPASS: "High - Requires authentication system redesign",
            VulnerabilityType.JWT_MANIPULATION: "Medium - Requires JWT implementation hardening",
            VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL: "High - Requires RBAC system review",
            VulnerabilityType.IDOR: "Low-Medium - Requires authorization check implementation",
        }

        vuln_types = {vuln.vulnerability_type for vuln in vulnerabilities}
        return {
            vuln_type.value: effort_mapping.get(vuln_type, "Medium - Requires investigation")
            for vuln_type in vuln_types
        }

    def _estimate_resource_requirements(
        self, vulnerabilities: List[VulnerabilityFinding]
    ) -> List[str]:
        """Estimate resource requirements for remediation."""
        requirements = set()

        vuln_types = {vuln.vulnerability_type for vuln in vulnerabilities}

        if any(
            t in vuln_types
            for t in [VulnerabilityType.AUTHENTICATION_BYPASS, VulnerabilityType.JWT_MANIPULATION]
        ):
            requirements.add("Security engineer with authentication expertise")

        if VulnerabilityType.SQL_INJECTION in vuln_types:
            requirements.add("Database security specialist")

        if any(
            t in vuln_types
            for t in [VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL, VulnerabilityType.IDOR]
        ):
            requirements.add("Application security developer")

        critical_count = sum(
            1 for vuln in vulnerabilities if vuln.severity == SeverityLevel.CRITICAL
        )
        if critical_count > 3:
            requirements.add("Dedicated security team for immediate response")

        return list(requirements)

    async def save_report(self, report: Dict[str, Any], file_path: Optional[str] = None) -> str:
        """Save penetration testing report to file."""
        if file_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            file_path = (
                f"/home/green/FreeAgentics/tests/security/reports/pentest_report_{timestamp}.json"
            )

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Penetration testing report saved to: {file_path}")
        return file_path


# Utility functions for common testing patterns


def generate_sql_injection_payloads() -> List[str]:
    """Generate common SQL injection payloads."""
    return [
        "' OR '1'='1",
        "' OR '1'='1' --",
        "' OR '1'='1' /*",
        "' UNION SELECT NULL,NULL,NULL --",
        "'; DROP TABLE users; --",
        "' AND SLEEP(5) --",
        "' AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --",
        "admin' --",
        "admin' #",
        "' OR username='admin' --",
        "1' OR '1'='1",
        "' OR 1=1 --",
        "' UNION ALL SELECT user(),database(),version() --",
        "' AND 1=CONVERT(int, (SELECT @@version)) --",
    ]


def generate_nosql_injection_payloads() -> List[Dict[str, Any]]:
    """Generate common NoSQL injection payloads."""
    return [
        {"$ne": None},
        {"$ne": ""},
        {"$regex": ".*"},
        {"$where": "1==1"},
        {"$exists": True},
        {"$gt": ""},
        {"$or": [{"username": "admin"}, {"username": {"$ne": ""}}]},
        {"username": {"$in": ["admin", "root", "administrator"]}},
        {"$or": [{"password": {"$exists": False}}, {"password": ""}]},
    ]


def generate_ldap_injection_payloads() -> List[str]:
    """Generate common LDAP injection payloads."""
    return [
        "*",
        "*)(&",
        "*))%00",
        ")(cn=*",
        "*()|%00",
        "*)(uid=*",
        "*)(objectClass=*",
        "admin*",
        "*)(mail=*",
        "*)(!(&(objectClass=*",
        "*)(!(mail=*))",
        "*)(!(cn=*))",
    ]


def generate_jwt_manipulation_payloads(original_token: str) -> List[str]:
    """Generate JWT manipulation payloads."""
    payloads = []

    try:
        # None algorithm attack
        import jwt

        header = jwt.get_unverified_header(original_token)
        payload = jwt.decode(original_token, options={"verify_signature": False})

        # Try "none" algorithm
        header["alg"] = "none"
        none_token = jwt.encode(payload, "", algorithm="none")
        payloads.append(none_token)

        # Try algorithm confusion (HS256 instead of RS256)
        if header.get("alg") == "RS256":
            header["alg"] = "HS256"
            hs256_token = jwt.encode(payload, "secret", algorithm="HS256")
            payloads.append(hs256_token)

        # Try role escalation
        if "role" in payload:
            payload["role"] = "admin"
            escalated_token = jwt.encode(payload, "fake_secret", algorithm="HS256")
            payloads.append(escalated_token)

        # Try extended expiration
        if "exp" in payload:
            payload["exp"] = payload["exp"] + 86400  # Add 24 hours
            extended_token = jwt.encode(payload, "fake_secret", algorithm="HS256")
            payloads.append(extended_token)

    except Exception as e:
        logger.warning(f"Error generating JWT manipulation payloads: {e}")

    return payloads


def generate_parameter_pollution_payloads() -> List[Dict[str, Any]]:
    """Generate parameter pollution attack payloads."""
    return [
        {"user_id": ["1", "2"]},  # Array pollution
        {"user_id": "1&user_id=2"},  # Query string pollution
        {"action": ["view", "delete"]},  # Action pollution
        {"role": ["user", "admin"]},  # Role pollution
        {"limit": ["10", "999999"]},  # Limit pollution
        {"offset": ["0", "-1"]},  # Offset pollution
    ]


if __name__ == "__main__":
    # Example usage
    framework = PenetrationTestingFramework()

    # Test modules will be registered in separate files
    # This is just the framework structure
    print("Penetration Testing Framework initialized")
    print("Register test modules and run with: await framework.run_all_tests()")
