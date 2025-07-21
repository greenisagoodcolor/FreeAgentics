"""
Authentication Error Disclosure Testing

Specialized tests for authentication-related error handling to ensure
no sensitive information is disclosed during authentication failures.

This module focuses on:
1. Username enumeration attacks
2. Password policy disclosure
3. Account lockout information leakage
4. Session management error disclosure
5. JWT error information leakage
6. Authentication timing attacks
"""

import json
import time
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.main import app


class AuthenticationErrorTester:
    """Specialized tester for authentication error disclosure."""

    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []

    def test_username_enumeration(self) -> Dict[str, Any]:
        """Test for username enumeration vulnerabilities."""
        results = {
            "test_name": "username_enumeration",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test different usernames - some exist, some don't
        test_usernames = [
            "admin",
            "administrator",
            "root",
            "user",
            "test",
            "nonexistent_user_12345",
            "another_fake_user_67890",
        ]

        response_times = {}
        response_messages = {}

        for username in test_usernames:
            start_time = time.time()

            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": username, "password": "wrong_password"},
            )

            end_time = time.time()
            response_time = end_time - start_time

            response_times[username] = response_time
            response_messages[username] = (
                response.json() if response.status_code != 500 else {"detail": "Server error"}
            )

        # Check for timing differences that could indicate username enumeration
        times = list(response_times.values())
        if times:
            avg_time = sum(times) / len(times)
            max_deviation = max(abs(t - avg_time) for t in times)

            # If timing differences are > 100ms, might indicate enumeration
            if max_deviation > 0.1:  # 100ms
                results["passed"] = False
                results["findings"].append(
                    {
                        "issue": "Potential timing-based username enumeration",
                        "max_deviation": max_deviation,
                        "response_times": response_times,
                    }
                )
                results["recommendations"].append(
                    "Implement consistent response timing for authentication failures"
                )

        # Check for different error messages that could indicate enumeration
        unique_messages = set()
        for username, message in response_messages.items():
            if isinstance(message, dict) and "detail" in message:
                unique_messages.add(message["detail"])

        if len(unique_messages) > 1:
            results["passed"] = False
            results["findings"].append(
                {
                    "issue": "Different error messages for authentication failures",
                    "unique_messages": list(unique_messages),
                    "response_messages": response_messages,
                }
            )
            results["recommendations"].append(
                "Use identical error messages for all authentication failures"
            )

        return results

    def test_password_policy_disclosure(self) -> Dict[str, Any]:
        """Test for password policy information disclosure."""
        results = {
            "test_name": "password_policy_disclosure",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test weak passwords to see if policy details are exposed
        weak_passwords = [
            "123",
            "password",
            "abc",
            "12345678",
            "qwerty",
            "admin",
            "",  # Empty password
            "a",  # Single character
            "12",  # Two characters
        ]

        for password in weak_passwords:
            response = self.client.post(
                "/api/v1/auth/register",
                json={
                    "username": "testuser_" + str(hash(password))[-6:],
                    "email": "test@example.com",
                    "password": password,
                    "role": "observer",
                },
            )

            if response.status_code == 400:  # Validation error
                response_data = response.json()
                response_text = json.dumps(response_data)

                # Check for specific policy disclosures
                policy_disclosures = [
                    "minimum.*length",
                    "maximum.*length",
                    "must.*contain.*uppercase",
                    "must.*contain.*lowercase",
                    "must.*contain.*digit",
                    "must.*contain.*special",
                    "cannot.*contain.*username",
                    "cannot.*be.*common",
                    "history.*check",
                    "entropy.*requirement",
                    "complexity.*score",
                ]

                for disclosure_pattern in policy_disclosures:
                    import re

                    if re.search(disclosure_pattern, response_text, re.IGNORECASE):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Password policy details disclosed",
                                "pattern": disclosure_pattern,
                                "password_tested": password,
                                "response": response_data,
                            }
                        )

        if results["findings"]:
            results["recommendations"].append(
                "Use generic password error messages without policy details"
            )
            results["recommendations"].append(
                "Provide password policy information separately, not in error messages"
            )

        return results

    def test_account_lockout_disclosure(self) -> Dict[str, Any]:
        """Test for account lockout information disclosure."""
        results = {
            "test_name": "account_lockout_disclosure",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Create a test user first
        test_username = "lockout_test_user"
        self.client.post(
            "/api/v1/auth/register",
            json={
                "username": test_username,
                "email": "lockout@example.com",
                "password": "ValidPassword123!",
                "role": "observer",
            },
        )

        # Attempt multiple failed logins
        for attempt in range(6):  # Try to trigger lockout
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": test_username, "password": "wrong_password"},
            )

            response_data = (
                response.json() if response.status_code != 500 else {"detail": "Server error"}
            )
            response_text = json.dumps(response_data)

            # Check for lockout-related disclosures
            lockout_disclosures = [
                "account.*locked",
                "too.*many.*attempts",
                r"try.*again.*in.*\d+",
                r"locked.*for.*\d+.*minutes",
                "attempts.*remaining",
                "lockout.*time",
                "unlock.*time",
                "retry.*after",
            ]

            for disclosure_pattern in lockout_disclosures:
                import re

                if re.search(disclosure_pattern, response_text, re.IGNORECASE):
                    results["passed"] = False
                    results["findings"].append(
                        {
                            "issue": "Account lockout information disclosed",
                            "pattern": disclosure_pattern,
                            "attempt": attempt + 1,
                            "response": response_data,
                        }
                    )

        if results["findings"]:
            results["recommendations"].append("Use generic error messages for account lockout")
            results["recommendations"].append(
                "Do not disclose lockout duration or remaining attempts"
            )

        return results

    def test_jwt_error_disclosure(self) -> Dict[str, Any]:
        """Test for JWT-related error information disclosure."""
        results = {
            "test_name": "jwt_error_disclosure",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test various invalid JWT scenarios
        invalid_tokens = [
            "invalid.jwt.token",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature",  # Invalid signature
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MDAwMDAwMDB9.signature",  # Expired
            "not.a.jwt",
            "",  # Empty token
            "Bearer invalid_token",
            "malformed_header",
        ]

        for token in invalid_tokens:
            headers = (
                {"Authorization": f"Bearer {token}"}
                if not token.startswith("Bearer")
                else {"Authorization": token}
            )

            response = self.client.get("/api/v1/agents", headers=headers)

            if response.status_code in [401, 403]:  # Auth errors
                response_data = (
                    response.json() if response.status_code != 500 else {"detail": "Server error"}
                )
                response_text = json.dumps(response_data)

                # Check for JWT-specific disclosures
                jwt_disclosures = [
                    "invalid.*signature",
                    "token.*expired",
                    "malformed.*token",
                    "algorithm.*mismatch",
                    "issuer.*invalid",
                    "audience.*invalid",
                    "not.*before.*claim",
                    "issued.*at.*claim",
                    "jwt.*decode.*error",
                    "cryptographic.*verification",
                    "key.*not.*found",
                    "header.*malformed",
                ]

                for disclosure_pattern in jwt_disclosures:
                    import re

                    if re.search(disclosure_pattern, response_text, re.IGNORECASE):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "JWT error details disclosed",
                                "pattern": disclosure_pattern,
                                "token_tested": token[:20] + "...",
                                "response": response_data,
                            }
                        )

        if results["findings"]:
            results["recommendations"].append('Use generic "Unauthorized" messages for JWT errors')
            results["recommendations"].append("Log JWT error details server-side only")

        return results

    def test_session_management_disclosure(self) -> Dict[str, Any]:
        """Test for session management error disclosure."""
        results = {
            "test_name": "session_management_disclosure",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Test session-related scenarios
        session_test_scenarios = [
            ("expired_session", "expired_session_id_12345"),
            ("invalid_session", "invalid_session_id_67890"),
            ("malformed_session", "malformed.session.id"),
            ("empty_session", ""),
            ("long_session", "a" * 1000),
        ]

        for scenario_name, session_id in session_test_scenarios:
            # Test with session cookie
            self.client.cookies.set("session_id", session_id)

            response = self.client.get("/api/v1/system/status")

            # Reset cookies
            self.client.cookies.clear()

            if response.status_code in [401, 403]:
                response_data = (
                    response.json() if response.status_code != 500 else {"detail": "Server error"}
                )
                response_text = json.dumps(response_data)

                # Check for session-specific disclosures
                session_disclosures = [
                    "session.*expired",
                    "session.*not.*found",
                    "session.*invalid",
                    "session.*timeout",
                    "session.*id.*malformed",
                    "session.*store.*error",
                    "redis.*connection.*failed",
                    "cache.*miss",
                    "session.*data.*corrupted",
                ]

                for disclosure_pattern in session_disclosures:
                    import re

                    if re.search(disclosure_pattern, response_text, re.IGNORECASE):
                        results["passed"] = False
                        results["findings"].append(
                            {
                                "issue": "Session management details disclosed",
                                "pattern": disclosure_pattern,
                                "scenario": scenario_name,
                                "response": response_data,
                            }
                        )

        if results["findings"]:
            results["recommendations"].append("Use generic authentication error messages")
            results["recommendations"].append("Do not expose session store implementation details")

        return results

    def test_authentication_timing_consistency(self) -> Dict[str, Any]:
        """Test authentication timing consistency to prevent timing attacks."""
        results = {
            "test_name": "authentication_timing_consistency",
            "passed": True,
            "findings": [],
            "recommendations": [],
        }

        # Create a known user for comparison
        known_user = "timing_test_user"
        self.client.post(
            "/api/v1/auth/register",
            json={
                "username": known_user,
                "email": "timing@example.com",
                "password": "ValidPassword123!",
                "role": "observer",
            },
        )

        # Test scenarios
        scenarios = [
            ("valid_user_wrong_password", known_user, "wrong_password"),
            (
                "invalid_user_any_password",
                "nonexistent_user_12345",
                "any_password",
            ),
            ("empty_user_empty_password", "", ""),
            ("long_user_long_password", "a" * 100, "b" * 100),
        ]

        timing_results = {}

        for scenario_name, username, password in scenarios:
            times = []

            # Run multiple attempts to get average timing
            for _ in range(5):
                start_time = time.time()

                self.client.post(
                    "/api/v1/auth/login",
                    json={"username": username, "password": password},
                )

                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            timing_results[scenario_name] = {
                "avg_time": avg_time,
                "times": times,
                "std_dev": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
            }

        # Check for significant timing differences
        all_avg_times = [data["avg_time"] for data in timing_results.values()]
        overall_avg = sum(all_avg_times) / len(all_avg_times)

        for scenario_name, data in timing_results.items():
            deviation = abs(data["avg_time"] - overall_avg)

            # If deviation is > 50ms, might indicate timing leak
            if deviation > 0.05:  # 50ms
                results["passed"] = False
                results["findings"].append(
                    {
                        "issue": "Significant timing difference detected",
                        "scenario": scenario_name,
                        "avg_time": data["avg_time"],
                        "overall_avg": overall_avg,
                        "deviation": deviation,
                    }
                )

        if results["findings"]:
            results["recommendations"].append("Implement constant-time authentication responses")
            results["recommendations"].append("Use artificial delays to normalize response times")
            results["recommendations"].append("Consider rate limiting to mitigate timing attacks")

        return results

    def run_all_authentication_tests(self) -> Dict[str, Any]:
        """Run all authentication error disclosure tests."""
        print("Running authentication error disclosure tests...")

        test_methods = [
            self.test_username_enumeration,
            self.test_password_policy_disclosure,
            self.test_account_lockout_disclosure,
            self.test_jwt_error_disclosure,
            self.test_session_management_disclosure,
            self.test_authentication_timing_consistency,
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

        # Collect all recommendations
        all_recommendations = []
        for result in all_results:
            all_recommendations.extend(result.get("recommendations", []))

        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            },
            "test_results": all_results,
            "recommendations": unique_recommendations,
            "overall_status": "PASS" if failed_tests == 0 else "FAIL",
        }

        return summary


class TestAuthenticationErrorDisclosure:
    """pytest test class for authentication error disclosure."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_tester(self, client):
        """Create authentication error tester."""
        return AuthenticationErrorTester(client)

    def test_username_enumeration_prevention(self, auth_tester):
        """Test that username enumeration is prevented."""
        result = auth_tester.test_username_enumeration()

        if not result["passed"]:
            failure_msg = "Username enumeration vulnerability detected:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}\n"
            pytest.fail(failure_msg)

    def test_password_policy_no_disclosure(self, auth_tester):
        """Test that password policy details are not disclosed."""
        result = auth_tester.test_password_policy_disclosure()

        if not result["passed"]:
            failure_msg = "Password policy disclosure detected:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}: {finding['pattern']}\n"
            pytest.fail(failure_msg)

    def test_account_lockout_no_disclosure(self, auth_tester):
        """Test that account lockout details are not disclosed."""
        result = auth_tester.test_account_lockout_disclosure()

        if not result["passed"]:
            failure_msg = "Account lockout information disclosure detected:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}: {finding['pattern']}\n"
            pytest.fail(failure_msg)

    def test_jwt_error_no_disclosure(self, auth_tester):
        """Test that JWT error details are not disclosed."""
        result = auth_tester.test_jwt_error_disclosure()

        if not result["passed"]:
            failure_msg = "JWT error information disclosure detected:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}: {finding['pattern']}\n"
            pytest.fail(failure_msg)

    def test_session_management_no_disclosure(self, auth_tester):
        """Test that session management details are not disclosed."""
        result = auth_tester.test_session_management_disclosure()

        if not result["passed"]:
            failure_msg = "Session management information disclosure detected:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}: {finding['pattern']}\n"
            pytest.fail(failure_msg)

    def test_authentication_timing_consistency(self, auth_tester):
        """Test that authentication timing is consistent."""
        result = auth_tester.test_authentication_timing_consistency()

        if not result["passed"]:
            failure_msg = "Authentication timing inconsistency detected:\n"
            for finding in result["findings"]:
                failure_msg += f"  - {finding['issue']}: {finding['scenario']} (deviation: {finding['deviation']:.3f}s)\n"
            pytest.fail(failure_msg)

    def test_comprehensive_authentication_security(self, auth_tester):
        """Run comprehensive authentication security tests."""
        summary = auth_tester.run_all_authentication_tests()

        if summary["overall_status"] == "FAIL":
            failure_msg = f"Authentication security test failures: {summary['summary']['failed_tests']} out of {summary['summary']['total_tests']} tests failed\n"

            for result in summary["test_results"]:
                if not result["passed"]:
                    failure_msg += f"\n{result['test_name']}:\n"
                    for finding in result["findings"]:
                        failure_msg += f"  - {finding['issue']}\n"

            if summary["recommendations"]:
                failure_msg += "\nRecommendations:\n"
                for rec in summary["recommendations"]:
                    failure_msg += f"  - {rec}\n"

            pytest.fail(failure_msg)


if __name__ == "__main__":
    """Direct execution for standalone testing."""
    client = TestClient(app)
    tester = AuthenticationErrorTester(client)

    print("Running authentication error disclosure tests...")
    summary = tester.run_all_authentication_tests()

    print(f"\n{'='*60}")
    print("AUTHENTICATION ERROR DISCLOSURE TEST REPORT")
    print(f"{'='*60}")
    print(f"Total Tests: {summary['summary']['total_tests']}")
    print(f"Passed: {summary['summary']['passed_tests']}")
    print(f"Failed: {summary['summary']['failed_tests']}")
    print(f"Pass Rate: {summary['summary']['pass_rate']:.1f}%")
    print(f"Overall Status: {summary['overall_status']}")

    if summary["recommendations"]:
        print(f"\n{'='*40}")
        print("RECOMMENDATIONS")
        print(f"{'='*40}")
        for rec in summary["recommendations"]:
            print(f"• {rec}")

    # Save report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = (
        f"/home/green/FreeAgentics/tests/security/auth_error_disclosure_report_{timestamp}.json"
    )

    try:
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Exit with appropriate code
    exit(0 if summary["overall_status"] == "PASS" else 1)
