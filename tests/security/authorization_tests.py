"""
Authorization Testing Module

This module implements comprehensive authorization testing including:
- Horizontal privilege escalation
- Vertical privilege escalation
- IDOR (Insecure Direct Object Reference) vulnerability testing
- Role-based access bypass
- Resource ownership validation
- Permission boundary testing
- Administrative function access control
"""

import logging
import time
from typing import List, Optional

from auth.security_implementation import UserRole

from .penetration_testing_framework import (
    BasePenetrationTest,
    SeverityLevel,
    TestResult,
    VulnerabilityFinding,
    VulnerabilityType,
)

logger = logging.getLogger(__name__)


class AuthorizationTests(BasePenetrationTest):
    """Comprehensive authorization vulnerability testing."""

    async def execute(self) -> TestResult:
        """Execute all authorization tests."""
        start_time = time.time()

        try:
            # Test horizontal privilege escalation
            await self._test_horizontal_privilege_escalation()

            # Test vertical privilege escalation
            await self._test_vertical_privilege_escalation()

            # Test IDOR vulnerabilities
            await self._test_idor_vulnerabilities()

            # Test role-based access bypass
            await self._test_role_bypass()

            # Test resource ownership validation
            await self._test_resource_ownership()

            # Test permission boundary enforcement
            await self._test_permission_boundaries()

            # Test administrative function access
            await self._test_admin_function_access()

            # Test API endpoint authorization
            await self._test_api_authorization()

            # Test parameter tampering for authorization bypass
            await self._test_parameter_tampering()

            # Test authorization token manipulation
            await self._test_token_privilege_escalation()

            execution_time = time.time() - start_time

            return TestResult(
                test_name="AuthorizationTests",
                success=True,
                vulnerabilities=self.vulnerabilities,
                execution_time=execution_time,
                metadata={
                    "test_users_created": self._count_test_users(),
                    "endpoints_tested": self._get_tested_endpoints(),
                    "roles_tested": [role.value for role in UserRole],
                },
            )

        except Exception as e:
            logger.error(f"Authorization test failed: {e}")
            return TestResult(
                test_name="AuthorizationTests",
                success=False,
                vulnerabilities=self.vulnerabilities,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def _test_horizontal_privilege_escalation(self):
        """Test horizontal privilege escalation vulnerabilities."""
        logger.info("Testing horizontal privilege escalation")

        # Create two users with same role
        user1_name, user1_pass, user1_id = self.create_test_user(UserRole.OBSERVER)
        user2_name, user2_pass, user2_id = self.create_test_user(UserRole.OBSERVER)

        # Get tokens for both users
        user1_token = self.get_auth_token(user1_name, user1_pass)
        self.get_auth_token(user2_name, user2_pass)

        # Test accessing user2's resources with user1's token
        test_cases = [
            # Try to access other user's profile/info
            (f"/api/v1/users/{user2_id}", "GET"),
            (f"/api/v1/users/{user2_id}/profile", "GET"),
            (f"/api/v1/users/{user2_id}/agents", "GET"),
            # Try to modify other user's data
            (f"/api/v1/users/{user2_id}", "PUT"),
            (f"/api/v1/users/{user2_id}/profile", "PATCH"),
        ]

        for endpoint, method in test_cases:
            response = self._make_request(method, endpoint, user1_token)

            if response.status_code in [200, 201, 202]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_HORIZONTAL,
                        severity=SeverityLevel.HIGH,
                        title=f"Horizontal Privilege Escalation in {endpoint}",
                        description=f"User {user1_id} can access resources belonging to user {user2_id}",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"{method} {endpoint}\nAuthorization: Bearer {user1_token[:20]}...\n"
                        f"Response: {response.status_code}",
                        exploitation_steps=[
                            "1. Authenticate as legitimate user",
                            "2. Identify resource IDs of other users",
                            f"3. Access {endpoint} with other user's ID",
                            "4. Gain unauthorized access to other user's data",
                        ],
                        remediation_steps=[
                            "Implement proper resource ownership validation",
                            "Check user ID against resource ownership before access",
                            "Use parameterized queries with user context",
                            "Implement access control lists (ACLs)",
                            "Log and monitor cross-user access attempts",
                        ],
                        cwe_id="CWE-639",
                        cvss_score=7.5,
                        test_method="horizontal_privilege_escalation",
                    )
                )

        # Test agent/resource enumeration
        await self._test_resource_enumeration(user1_token, user1_id, user2_id)

    async def _test_vertical_privilege_escalation(self):
        """Test vertical privilege escalation vulnerabilities."""
        logger.info("Testing vertical privilege escalation")

        # Create users with different privilege levels
        observer_name, observer_pass, observer_id = self.create_test_user(
            UserRole.OBSERVER
        )
        (
            researcher_name,
            researcher_pass,
            researcher_id,
        ) = self.create_test_user(UserRole.RESEARCHER)
        admin_name, admin_pass, admin_id = self.create_test_user(UserRole.ADMIN)

        # Get tokens
        observer_token = self.get_auth_token(observer_name, observer_pass)
        self.get_auth_token(researcher_name, researcher_pass)

        # Test observer trying to access researcher/admin functions
        privileged_endpoints = [
            # Admin-only endpoints
            ("/api/v1/admin/users", "GET"),
            ("/api/v1/admin/system", "GET"),
            ("/api/v1/admin/logs", "GET"),
            ("/api/v1/users", "POST"),  # Create user
            ("/api/v1/system/shutdown", "POST"),
            # Researcher-level endpoints
            ("/api/v1/agents", "POST"),  # Create agent
            ("/api/v1/agents/123", "DELETE"),  # Delete agent
            ("/api/v1/coalitions", "POST"),  # Create coalition
        ]

        for endpoint, method in privileged_endpoints:
            # Test with observer token (lowest privilege)
            response = self._make_request(method, endpoint, observer_token)

            if response.status_code in [200, 201, 202]:
                severity = (
                    SeverityLevel.CRITICAL
                    if "admin" in endpoint
                    else SeverityLevel.HIGH
                )

                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                        severity=severity,
                        title=f"Vertical Privilege Escalation in {endpoint}",
                        description=f"Observer role can access privileged endpoint: {endpoint}",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"{method} {endpoint}\nAuthorization: Bearer {observer_token[:20]}...\n"
                        f"Response: {response.status_code}",
                        exploitation_steps=[
                            "1. Authenticate with low-privilege account (Observer)",
                            f"2. Access privileged endpoint: {method} {endpoint}",
                            "3. Gain unauthorized access to privileged functions",
                            "4. Potentially escalate to administrative access",
                        ],
                        remediation_steps=[
                            "Implement proper role-based access control (RBAC)",
                            "Validate user permissions for each endpoint",
                            "Use principle of least privilege",
                            "Implement defense-in-depth authorization checks",
                            "Regular audit of endpoint permissions",
                        ],
                        cwe_id="CWE-269",
                        cvss_score=9.1 if severity == SeverityLevel.CRITICAL else 8.5,
                        test_method="vertical_privilege_escalation",
                    )
                )

        # Test privilege escalation through parameter manipulation
        await self._test_role_parameter_manipulation(observer_token)

    async def _test_idor_vulnerabilities(self):
        """Test Insecure Direct Object Reference vulnerabilities."""
        logger.info("Testing IDOR vulnerabilities")

        # Create test users
        user1_name, user1_pass, user1_id = self.create_test_user(UserRole.RESEARCHER)
        user2_name, user2_pass, user2_id = self.create_test_user(UserRole.RESEARCHER)

        user1_token = self.get_auth_token(user1_name, user1_pass)

        # Test numeric ID manipulation
        await self._test_numeric_idor(user1_token, user1_id, user2_id)

        # Test UUID manipulation (if applicable)
        await self._test_uuid_idor(user1_token, user1_id, user2_id)

        # Test sequential ID enumeration
        await self._test_sequential_enumeration(user1_token)

        # Test file/resource IDOR
        await self._test_file_idor(user1_token)

    async def _test_role_bypass(self):
        """Test role-based access control bypass."""
        logger.info("Testing role bypass vulnerabilities")

        # Create user with limited role
        user_name, user_pass, user_id = self.create_test_user(UserRole.OBSERVER)
        user_token = self.get_auth_token(user_name, user_pass)

        # Test role bypass techniques
        bypass_techniques = [
            # HTTP method override
            ("X-HTTP-Method-Override", "POST"),
            ("X-HTTP-Method-Override", "DELETE"),
            ("X-Method-Override", "POST"),
            # Request smuggling attempts
            ("Transfer-Encoding", "chunked"),
            # Parameter pollution
            ("role", "admin"),
            ("user_role", "admin"),
            ("privilege", "admin"),
        ]

        for header_name, header_value in bypass_techniques:
            headers = self.get_auth_headers(user_token)
            headers[header_name] = header_value

            # Test on privileged endpoint
            response = self.client.post(
                "/api/v1/agents",
                json={"name": "test_agent", "type": "basic"},
                headers=headers,
            )

            if response.status_code in [200, 201]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                        severity=SeverityLevel.HIGH,
                        title=f"Role Bypass via {header_name} Header",
                        description=f"Role restrictions bypassed using {header_name}: {header_value}",
                        affected_endpoint="/api/v1/agents",
                        proof_of_concept=f"POST /api/v1/agents\n{header_name}: {header_value}\n"
                        f"Authorization: Bearer {user_token[:20]}...",
                        exploitation_steps=[
                            "1. Authenticate with limited-privilege account",
                            f"2. Add header: {header_name}: {header_value}",
                            "3. Access privileged endpoint",
                            "4. Bypass role-based access controls",
                        ],
                        remediation_steps=[
                            "Implement server-side authorization checks",
                            "Ignore client-provided role/privilege headers",
                            "Validate authorization on every request",
                            "Use server-side session for role information",
                        ],
                        cwe_id="CWE-863",
                        cvss_score=7.5,
                        test_method="role_bypass",
                    )
                )

    async def _test_resource_ownership(self):
        """Test resource ownership validation."""
        logger.info("Testing resource ownership validation")

        # Create users and resources
        owner_name, owner_pass, owner_id = self.create_test_user(UserRole.RESEARCHER)
        attacker_name, attacker_pass, attacker_id = self.create_test_user(
            UserRole.RESEARCHER
        )

        owner_token = self.get_auth_token(owner_name, owner_pass)
        attacker_token = self.get_auth_token(attacker_name, attacker_pass)

        # Create a resource as owner
        create_response = self.client.post(
            "/api/v1/agents",
            json={"name": "owned_agent", "type": "basic"},
            headers=self.get_auth_headers(owner_token),
        )

        if create_response.status_code in [200, 201]:
            try:
                agent_data = create_response.json()
                agent_id = agent_data.get("id") or agent_data.get("agent_id")

                if agent_id:
                    # Try to access/modify the resource with different user
                    ownership_tests = [
                        (f"/api/v1/agents/{agent_id}", "GET"),
                        (f"/api/v1/agents/{agent_id}", "PUT"),
                        (f"/api/v1/agents/{agent_id}", "DELETE"),
                        (f"/api/v1/agents/{agent_id}/config", "PATCH"),
                    ]

                    for endpoint, method in ownership_tests:
                        response = self._make_request(method, endpoint, attacker_token)

                        if response.status_code in [200, 201, 202, 204]:
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.IDOR,
                                    severity=SeverityLevel.HIGH,
                                    title=f"Resource Ownership Bypass in {endpoint}",
                                    description=f"User {attacker_id} can access resource owned by {owner_id}",
                                    affected_endpoint=endpoint,
                                    proof_of_concept=f"{method} {endpoint}\nResource owner: {owner_id}\n"
                                    f"Accessing user: {attacker_id}",
                                    exploitation_steps=[
                                        "1. Identify resources owned by other users",
                                        "2. Use resource IDs to access endpoints",
                                        "3. Perform unauthorized operations on resources",
                                        "4. Potentially modify or delete other users' data",
                                    ],
                                    remediation_steps=[
                                        "Implement resource ownership validation",
                                        "Check user ID against resource owner before operations",
                                        "Use ACLs for resource access control",
                                        "Implement proper database queries with user context",
                                    ],
                                    cwe_id="CWE-639",
                                    cvss_score=8.1,
                                    test_method="resource_ownership",
                                )
                            )

            except Exception as e:
                logger.debug(f"Resource ownership test error: {e}")

    async def _test_permission_boundaries(self):
        """Test permission boundary enforcement."""
        logger.info("Testing permission boundaries")

        # Create user with specific limited permissions
        user_name, user_pass, user_id = self.create_test_user(UserRole.AGENT_MANAGER)
        user_token = self.get_auth_token(user_name, user_pass)

        # Test operations outside of granted permissions
        # AGENT_MANAGER should not have DELETE_AGENT or ADMIN_SYSTEM permissions
        forbidden_operations = [
            ("/api/v1/agents/123", "DELETE"),  # DELETE_AGENT permission needed
            ("/api/v1/admin/system", "GET"),  # ADMIN_SYSTEM permission needed
            ("/api/v1/admin/users", "GET"),  # ADMIN_SYSTEM permission needed
            ("/api/v1/system/config", "PUT"),  # ADMIN_SYSTEM permission needed
        ]

        for endpoint, method in forbidden_operations:
            response = self._make_request(method, endpoint, user_token)

            if response.status_code in [200, 201, 202, 204]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                        severity=SeverityLevel.HIGH,
                        title=f"Permission Boundary Violation in {endpoint}",
                        description="AGENT_MANAGER role accessed operation requiring higher permissions",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"{method} {endpoint}\nUser role: AGENT_MANAGER\n"
                        f"Required permission: Not granted",
                        exploitation_steps=[
                            "1. Authenticate with limited-permission account",
                            "2. Access endpoints requiring higher permissions",
                            "3. Perform unauthorized administrative operations",
                        ],
                        remediation_steps=[
                            "Implement fine-grained permission checking",
                            "Map each endpoint to required permissions",
                            "Validate user permissions on every request",
                            "Use permission-based access control (PBAC)",
                        ],
                        cwe_id="CWE-862",
                        cvss_score=7.8,
                        test_method="permission_boundaries",
                    )
                )

    async def _test_admin_function_access(self):
        """Test administrative function access control."""
        logger.info("Testing admin function access control")

        # Create non-admin user
        user_name, user_pass, user_id = self.create_test_user(UserRole.RESEARCHER)
        user_token = self.get_auth_token(user_name, user_pass)

        # Test access to admin functions
        admin_functions = [
            ("/api/v1/admin/users", "GET"),
            ("/api/v1/admin/users", "POST"),
            ("/api/v1/admin/users/123", "DELETE"),
            ("/api/v1/admin/system/config", "GET"),
            ("/api/v1/admin/system/config", "PUT"),
            ("/api/v1/admin/logs", "GET"),
            ("/api/v1/admin/metrics", "GET"),
            ("/api/v1/system/shutdown", "POST"),
            ("/api/v1/system/restart", "POST"),
        ]

        for endpoint, method in admin_functions:
            response = self._make_request(method, endpoint, user_token)

            if response.status_code in [200, 201, 202, 204]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                        severity=SeverityLevel.CRITICAL,
                        title=f"Unauthorized Admin Access: {endpoint}",
                        description=f"Non-admin user can access administrative function: {endpoint}",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"{method} {endpoint}\nUser role: RESEARCHER\n"
                        f"Response: {response.status_code}",
                        exploitation_steps=[
                            "1. Authenticate with non-administrative account",
                            "2. Access administrative endpoints",
                            "3. Perform privileged operations",
                            "4. Potentially compromise entire system",
                        ],
                        remediation_steps=[
                            "Implement strict admin role validation",
                            "Use separate authentication for admin functions",
                            "Implement IP-based restrictions for admin access",
                            "Add multi-factor authentication for admin operations",
                            "Log and monitor all admin function access",
                        ],
                        cwe_id="CWE-269",
                        cvss_score=9.8,
                        test_method="admin_function_access",
                    )
                )

    async def _test_api_authorization(self):
        """Test API endpoint authorization consistently."""
        logger.info("Testing API authorization consistency")

        # Test unauthorized access (no token)
        endpoints_to_test = [
            ("/api/v1/agents", "GET"),
            ("/api/v1/agents", "POST"),
            ("/api/v1/agents/123", "GET"),
            ("/api/v1/agents/123", "PUT"),
            ("/api/v1/agents/123", "DELETE"),
            ("/api/v1/auth/me", "GET"),
            ("/api/v1/auth/permissions", "GET"),
        ]

        for endpoint, method in endpoints_to_test:
            response = self._make_request(method, endpoint, None)  # No token

            if response.status_code in [200, 201, 202]:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                        severity=SeverityLevel.HIGH,
                        title=f"Missing Authorization in {endpoint}",
                        description=f"Endpoint {endpoint} accessible without authentication",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"{method} {endpoint}\nNo Authorization header\n"
                        f"Response: {response.status_code}",
                        exploitation_steps=[
                            f"1. Send {method} request to {endpoint}",
                            "2. No authentication required",
                            "3. Gain unauthorized access to protected resources",
                        ],
                        remediation_steps=[
                            "Implement authentication checks on all protected endpoints",
                            "Use authentication middleware",
                            "Default to deny access for unauthenticated requests",
                            "Regularly audit endpoint protection",
                        ],
                        cwe_id="CWE-306",
                        cvss_score=7.5,
                        test_method="api_authorization",
                    )
                )

    async def _test_parameter_tampering(self):
        """Test authorization bypass through parameter tampering."""
        logger.info("Testing parameter tampering for authorization bypass")

        # Create limited user
        user_name, user_pass, user_id = self.create_test_user(UserRole.OBSERVER)
        user_token = self.get_auth_token(user_name, user_pass)

        # Test parameter injection for privilege escalation
        tampered_requests = [
            # Try to set admin role in request
            {
                "endpoint": "/api/v1/agents",
                "method": "POST",
                "data": {
                    "name": "test",
                    "role": "admin",
                    "permissions": ["admin"],
                },
                "description": "Role injection in request body",
            },
            # Try user_id tampering
            {
                "endpoint": "/api/v1/auth/me",
                "method": "GET",
                "params": {"user_id": "admin", "as_user": "admin"},
                "description": "User ID tampering in parameters",
            },
            # Try permission escalation via parameters
            {
                "endpoint": "/api/v1/agents",
                "method": "POST",
                "data": {
                    "name": "test",
                    "owner": "admin",
                    "created_by": "admin",
                },
                "description": "Owner tampering in request",
            },
        ]

        for request_info in tampered_requests:
            if request_info["method"] == "POST":
                response = self.client.post(
                    request_info["endpoint"],
                    json=request_info.get("data", {}),
                    params=request_info.get("params", {}),
                    headers=self.get_auth_headers(user_token),
                )
            else:
                response = self.client.get(
                    request_info["endpoint"],
                    params=request_info.get("params", {}),
                    headers=self.get_auth_headers(user_token),
                )

            if response.status_code in [
                200,
                201,
            ] and self._detect_privilege_escalation(response):
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                        severity=SeverityLevel.HIGH,
                        title="Parameter Tampering Authorization Bypass",
                        description=f"{request_info['description']} allows authorization bypass",
                        affected_endpoint=request_info["endpoint"],
                        proof_of_concept=f"{request_info['method']} {request_info['endpoint']}\n"
                        f"Data: {request_info.get('data', {})}\n"
                        f"Params: {request_info.get('params', {})}",
                        exploitation_steps=[
                            "1. Authenticate with limited privileges",
                            "2. Tamper with request parameters",
                            "3. Include admin/privileged values in request",
                            "4. Gain elevated access or bypass restrictions",
                        ],
                        remediation_steps=[
                            "Validate all input parameters server-side",
                            "Ignore client-provided privilege/role parameters",
                            "Use server-side session for authorization context",
                            "Implement parameter validation and sanitization",
                        ],
                        cwe_id="CWE-20",
                        cvss_score=7.5,
                        test_method="parameter_tampering",
                    )
                )

    async def _test_token_privilege_escalation(self):
        """Test privilege escalation through token manipulation."""
        logger.info("Testing token privilege escalation")

        # Create test user
        user_name, user_pass, user_id = self.create_test_user(UserRole.OBSERVER)
        original_token = self.get_auth_token(user_name, user_pass)

        # Try to manipulate JWT token for privilege escalation
        try:
            import jwt as pyjwt

            # Decode token without verification
            payload = pyjwt.decode(original_token, options={"verify_signature": False})

            # Try role escalation
            escalated_payload = payload.copy()
            escalated_payload["role"] = "admin"
            escalated_payload["permissions"] = [
                "admin_system",
                "create_agent",
                "delete_agent",
            ]

            # Try different signing methods
            escalation_attempts = [
                ("none", ""),
                ("HS256", "secret"),
                ("HS256", "admin"),
                ("HS256", "password"),
            ]

            for algorithm, secret in escalation_attempts:
                try:
                    escalated_token = pyjwt.encode(
                        escalated_payload, secret, algorithm=algorithm
                    )

                    # Test escalated token
                    response = self.client.get(
                        "/api/v1/admin/users",
                        headers={"Authorization": f"Bearer {escalated_token}"},
                    )

                    if response.status_code == 200:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.JWT_MANIPULATION,
                                severity=SeverityLevel.CRITICAL,
                                title=f"JWT Token Privilege Escalation - {algorithm}",
                                description=f"JWT token manipulation allows privilege escalation using {algorithm}",
                                affected_endpoint="/api/v1/admin/users",
                                proof_of_concept=f"Original role: {payload.get('role')}\n"
                                f"Escalated role: admin\n"
                                f"Algorithm: {algorithm}\nSecret: {secret}",
                                exploitation_steps=[
                                    "1. Obtain valid JWT token",
                                    "2. Decode token payload",
                                    "3. Modify role and permissions",
                                    f"4. Re-sign with {algorithm} algorithm",
                                    "5. Access administrative functions",
                                ],
                                remediation_steps=[
                                    "Use strong, random secret keys",
                                    "Implement proper algorithm validation",
                                    "Use asymmetric algorithms (RS256) instead of symmetric",
                                    "Validate token claims server-side",
                                    "Implement token binding and validation",
                                ],
                                cwe_id="CWE-345",
                                cvss_score=9.8,
                                test_method="token_privilege_escalation",
                            )
                        )

                except Exception as e:
                    logger.debug(f"Token escalation attempt failed: {e}")

        except Exception as e:
            logger.debug(f"Token manipulation test error: {e}")

    # Helper methods for IDOR testing

    async def _test_numeric_idor(self, token: str, user1_id: str, user2_id: str):
        """Test numeric ID manipulation for IDOR."""
        # Try sequential IDs around known user ID
        try:
            base_id = int(user1_id) if user1_id.isdigit() else hash(user1_id) % 10000

            for test_id in range(base_id - 5, base_id + 5):
                if str(test_id) != user1_id:  # Don't test own ID
                    endpoint = f"/api/v1/users/{test_id}"
                    response = self.client.get(
                        endpoint, headers=self.get_auth_headers(token)
                    )

                    if response.status_code == 200:
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.IDOR,
                                severity=SeverityLevel.MEDIUM,
                                title="Numeric IDOR in User Access",
                                description=f"Can access user {test_id} data through ID manipulation",
                                affected_endpoint=endpoint,
                                proof_of_concept=f"GET {endpoint} returns user data",
                                exploitation_steps=[
                                    "1. Identify numeric user IDs",
                                    "2. Enumerate sequential IDs",
                                    "3. Access other users' data",
                                ],
                                remediation_steps=[
                                    "Implement authorization checks",
                                    "Use UUIDs instead of sequential IDs",
                                    "Validate resource ownership",
                                ],
                                cwe_id="CWE-639",
                                cvss_score=6.5,
                                test_method="numeric_idor",
                            )
                        )
                        break  # Only report first finding

        except Exception as e:
            logger.debug(f"Numeric IDOR test error: {e}")

    async def _test_uuid_idor(self, token: str, user1_id: str, user2_id: str):
        """Test UUID manipulation for IDOR."""
        if len(user2_id) > 10:  # Likely UUID
            endpoint = f"/api/v1/users/{user2_id}"
            response = self.client.get(endpoint, headers=self.get_auth_headers(token))

            if response.status_code == 200:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.IDOR,
                        severity=SeverityLevel.HIGH,
                        title="UUID IDOR in User Access",
                        description=f"Can access user {user2_id} data",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"GET {endpoint} returns other user's data",
                        exploitation_steps=[
                            "1. Obtain valid user UUIDs",
                            "2. Use UUIDs to access other users' data",
                            "3. Enumerate user information",
                        ],
                        remediation_steps=[
                            "Implement proper authorization checks",
                            "Validate user owns requested resource",
                            "Use ACLs for resource access",
                        ],
                        cwe_id="CWE-639",
                        cvss_score=7.5,
                        test_method="uuid_idor",
                    )
                )

    async def _test_sequential_enumeration(self, token: str):
        """Test sequential ID enumeration."""
        # Test common sequential patterns
        for base_id in [1, 100, 1000]:
            for offset in range(5):
                test_id = base_id + offset
                endpoint = f"/api/v1/agents/{test_id}"
                response = self.client.get(
                    endpoint, headers=self.get_auth_headers(token)
                )

                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data and "id" in data:
                            self.add_vulnerability(
                                VulnerabilityFinding(
                                    vulnerability_type=VulnerabilityType.IDOR,
                                    severity=SeverityLevel.MEDIUM,
                                    title="Sequential ID Enumeration",
                                    description="Sequential IDs allow resource enumeration",
                                    affected_endpoint=endpoint,
                                    proof_of_concept=f"GET {endpoint} returns: {data.get('id')}",
                                    exploitation_steps=[
                                        "1. Identify sequential ID pattern",
                                        "2. Enumerate resources by incrementing IDs",
                                        "3. Access unauthorized resources",
                                    ],
                                    remediation_steps=[
                                        "Use non-sequential UUIDs",
                                        "Implement proper access controls",
                                        "Add rate limiting to prevent enumeration",
                                    ],
                                    cwe_id="CWE-639",
                                    cvss_score=5.5,
                                    test_method="sequential_enumeration",
                                )
                            )
                            break
                    except Exception:
                        pass

    async def _test_file_idor(self, token: str):
        """Test file/document IDOR vulnerabilities."""
        # Test common file access patterns
        file_endpoints = [
            "/api/v1/files/1",
            "/api/v1/documents/1",
            "/api/v1/reports/1",
            "/api/v1/uploads/1",
            "/api/v1/attachments/1",
        ]

        for endpoint in file_endpoints:
            response = self.client.get(endpoint, headers=self.get_auth_headers(token))

            if response.status_code == 200:
                self.add_vulnerability(
                    VulnerabilityFinding(
                        vulnerability_type=VulnerabilityType.IDOR,
                        severity=SeverityLevel.MEDIUM,
                        title=f"File IDOR in {endpoint}",
                        description="Can access files without proper authorization",
                        affected_endpoint=endpoint,
                        proof_of_concept=f"GET {endpoint} returns file content",
                        exploitation_steps=[
                            "1. Identify file access endpoints",
                            "2. Enumerate file IDs",
                            "3. Access unauthorized files",
                        ],
                        remediation_steps=[
                            "Implement file ownership validation",
                            "Use access control for file downloads",
                            "Validate user permissions for file access",
                        ],
                        cwe_id="CWE-639",
                        cvss_score=6.1,
                        test_method="file_idor",
                    )
                )

    async def _test_resource_enumeration(
        self, token: str, user1_id: str, user2_id: str
    ):
        """Test resource enumeration capabilities."""
        # Test if user can enumerate other users' resources
        enum_endpoints = [
            f"/api/v1/users/{user2_id}/agents",
            f"/api/v1/users/{user2_id}/coalitions",
            f"/api/v1/users/{user2_id}/projects",
        ]

        for endpoint in enum_endpoints:
            response = self.client.get(endpoint, headers=self.get_auth_headers(token))

            if response.status_code == 200:
                try:
                    data = response.json()
                    if data and (isinstance(data, list) or "items" in data):
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_HORIZONTAL,
                                severity=SeverityLevel.MEDIUM,
                                title=f"Resource Enumeration in {endpoint}",
                                description=f"Can enumerate user {user2_id}'s resources",
                                affected_endpoint=endpoint,
                                proof_of_concept=f"GET {endpoint} returns resource list",
                                exploitation_steps=[
                                    "1. Obtain other users' IDs",
                                    "2. Enumerate their resources",
                                    "3. Gain intelligence about other users' activities",
                                ],
                                remediation_steps=[
                                    "Implement resource visibility controls",
                                    "Validate user permissions for resource listing",
                                    "Use privacy controls for user data",
                                ],
                                cwe_id="CWE-639",
                                cvss_score=4.3,
                                test_method="resource_enumeration",
                            )
                        )
                except Exception:
                    pass

    async def _test_role_parameter_manipulation(self, token: str):
        """Test role escalation through parameter manipulation."""
        role_params = [
            {"role": "admin"},
            {"user_role": "admin"},
            {"privilege": "admin"},
            {"permission": "admin"},
            {"level": "admin"},
            {"access_level": "admin"},
        ]

        for params in role_params:
            response = self.client.get(
                "/api/v1/auth/me",
                params=params,
                headers=self.get_auth_headers(token),
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("role") == "admin" or "admin" in str(
                        data.get("permissions", [])
                    ):
                        self.add_vulnerability(
                            VulnerabilityFinding(
                                vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION_VERTICAL,
                                severity=SeverityLevel.CRITICAL,
                                title="Role Escalation via Parameter Manipulation",
                                description="Role escalated through URL parameters",
                                affected_endpoint="/api/v1/auth/me",
                                proof_of_concept=f"GET /api/v1/auth/me?{list(params.keys())[0]}=admin",
                                exploitation_steps=[
                                    "1. Add role parameter to request",
                                    "2. Set parameter to admin value",
                                    "3. Gain administrative privileges",
                                ],
                                remediation_steps=[
                                    "Ignore client-provided role parameters",
                                    "Use server-side session for role information",
                                    "Validate all input parameters",
                                ],
                                cwe_id="CWE-20",
                                cvss_score=9.1,
                                test_method="role_parameter_manipulation",
                            )
                        )
                except Exception:
                    pass

    # Helper methods

    def _make_request(self, method: str, endpoint: str, token: Optional[str], **kwargs):
        """Make HTTP request with optional authentication."""
        headers = self.get_auth_headers(token) if token else {}
        headers.update(kwargs.get("headers", {}))

        if method == "GET":
            return self.client.get(endpoint, headers=headers, **kwargs)
        elif method == "POST":
            return self.client.post(endpoint, headers=headers, **kwargs)
        elif method == "PUT":
            return self.client.put(endpoint, headers=headers, **kwargs)
        elif method == "PATCH":
            return self.client.patch(endpoint, headers=headers, **kwargs)
        elif method == "DELETE":
            return self.client.delete(endpoint, headers=headers, **kwargs)
        else:
            return self.client.get(endpoint, headers=headers, **kwargs)

    def _detect_privilege_escalation(self, response) -> bool:
        """Detect if response indicates privilege escalation."""
        try:
            data = response.json() if response.text else {}
            response_text = response.text.lower()

            # Check for admin indicators
            admin_indicators = ["admin", "administrator", "root", "superuser"]
            privilege_indicators = [
                "create_agent",
                "delete_agent",
                "admin_system",
            ]

            return any(
                indicator in response_text for indicator in admin_indicators
            ) or any(indicator in str(data) for indicator in privilege_indicators)
        except Exception:
            return False

    def _count_test_users(self) -> int:
        """Count the number of test users created."""
        return len(
            [
                user
                for user in self.auth_manager.users.keys()
                if user.startswith("test_user_")
            ]
        )

    def _get_tested_endpoints(self) -> List[str]:
        """Get list of endpoints tested."""
        return [
            "/api/v1/users/{id}",
            "/api/v1/agents",
            "/api/v1/agents/{id}",
            "/api/v1/admin/users",
            "/api/v1/admin/system",
            "/api/v1/auth/me",
            "/api/v1/auth/permissions",
        ]
